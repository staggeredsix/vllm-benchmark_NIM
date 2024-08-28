import os
import subprocess
import logging
import time
import socket
import asyncio
from vllm_benchmark import run_benchmark  # Ensure correct import path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

NGC_KEY_ENV_VAR = "NGC_API_KEY"
NIM_LIST_FILE = "nim_list.txt"
PORT = 8000

def input_ngc_key():
    ngc_key = input("Enter your NGC API key: ").strip()
    os.environ[NGC_KEY_ENV_VAR] = ngc_key
    save_ngc_key_to_file(ngc_key)
    print("NGC API key has been set and saved.")

def save_ngc_key_to_file(ngc_key):
    with open(".ngc_api_key", "w") as f:
        f.write(ngc_key)

def load_ngc_key_from_file():
    if os.path.exists(".ngc_api_key"):
        with open(".ngc_api_key", "r") as f:
            ngc_key = f.read().strip()
            os.environ[NGC_KEY_ENV_VAR] = ngc_key
            return ngc_key
    return None

def change_ngc_key():
    if os.path.exists(".ngc_api_key"):
        os.remove(".ngc_api_key")
    input_ngc_key()

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

def stop_all_containers():
    print("Killing all running containers and purging them...")
    subprocess.run("docker stop $(docker ps -q)", shell=True)
    subprocess.run("docker rm $(docker ps -a -q)", shell=True)
    subprocess.run("docker container prune -f", shell=True)
    print("All containers have been killed and purged.")

def start_container(img_name, gpus, local_nim_cache):
    if not gpus:
        print("Error: You must specify the GPU indices (e.g., '0', '0,1', 'all').")
        return None

    gpus_flag = "--gpus=all" if gpus.lower() == "all" else f"--gpus=\"device={gpus}\""

    run_command = (
        f"docker run --rm {gpus_flag} --shm-size=16GB "
        f"-e {NGC_KEY_ENV_VAR}={os.environ[NGC_KEY_ENV_VAR]} -v {local_nim_cache}:/opt/nim/.cache "
        f"-u {os.getuid()}:{os.getgid()} -p {PORT}:8000 {img_name}"
    )

    logging.info(f"Running Docker command: {run_command}")
    print(f"Running Docker command: {run_command}")

    container_process = subprocess.Popen(run_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print("Container started successfully.")
    return container_process

def wait_for_uvicorn(container_process):
    print("Waiting for Uvicorn server to start...")
    while True:
        line = container_process.stdout.readline().decode('utf-8').strip()
        if line:
            print(line)
        if "Uvicorn running on" in line:
            print("Uvicorn server is running.")
            break

async def auto_scale_benchmark(base_url, initial_concurrency, model_name, api_key, tps_threshold=12):
    concurrency = initial_concurrency
    total_requests = concurrency * 5  # Starting with 5x the concurrency level

    while True:
        logging.info(f"Running with concurrency {concurrency} and total requests {total_requests}")
        model_url = f"{base_url}/v1/"
        
        response = await run_benchmark(total_requests, concurrency, 30, 100, model_url, api_key, False, model_name)
        
        if "error" in response:
            logging.error(f"Error during benchmark: {response['error'].get('message', 'Unknown error')}")
            break
        
        tps = response.get("tokens_per_second", {}).get("average", None)
        
        if tps is None:
            logging.error("Failed to retrieve TPS from response data.")
            break
        
        if tps < tps_threshold:
            print(f"Test stopped as TPS dropped below {tps_threshold} with concurrency {concurrency}.")
            break

        # Adjust the concurrency and total requests based on the TPS
        if tps > 100:
            concurrency *= 2
        elif tps > 50:
            concurrency += int(concurrency * 0.5)
        else:
            concurrency += int(concurrency * 0.2)
        
        total_requests = concurrency * 5

async def run_benchmark_tests(base_url, total_requests, concurrency_level, model_name):
    model_url = f"{base_url}/v1/"
    
    response = await run_benchmark(total_requests, concurrency_level, 30, 100, model_url, os.environ[NGC_KEY_ENV_VAR], False, model_name)
    
    if "error" in response:
        logging.error(f"Error during benchmark: {response['error'].get('message', 'Unknown error')}")
    else:
        logging.info(f"Benchmark completed successfully: {response}")

def load_nims_from_file():
    nims = []
    if os.path.exists(NIM_LIST_FILE):
        with open(NIM_LIST_FILE, "r") as f:
            for line in f:
                if "|" in line:
                    nim_name, img_name = line.strip().split("|")
                    model_name = img_name.split("/")[-1].split(":")[0]
                    developer_name = img_name.split("/")[-2]
                    full_model_name = f"{developer_name}/{model_name}"
                    nims.append([full_model_name, img_name])
    return nims

def save_nims_to_file(nims):
    with open(NIM_LIST_FILE, "w") as f:
        for nim in nims:
            f.write(f"{nim[0]}|{nim[1]}\n")

def add_nim():
    nim_name = input("Enter the NIM name (e.g., mistral-7b-instruct): ").strip()
    img_name = input("Enter the Docker image name (e.g., nvcr.io/nim/meta/llama3-8b-instruct:latest): ").strip()
    nims = load_nims_from_file()
    model_name = img_name.split("/")[-1].split(":")[0]
    developer_name = img_name.split("/")[-2]
    full_model_name = f"{developer_name}/{model_name}"
    nims.append([full_model_name, img_name])
    save_nims_to_file(nims)
    print(f"NIM {nim_name} added.")

def list_nims():
    nims = load_nims_from_file()
    if not nims:
        print("No NIMs found.")
    else:
        print("\nAvailable NIMs:")
        for idx, nim in enumerate(nims):
            print(f"{idx + 1}. {nim[0]} ({nim[1]})")

async def run_tests_menu():
    nims = load_nims_from_file()
    if not nims:
        print("No NIMs found. Please add a NIM first.")
        return

    list_nims()
    nim_choice = int(input("Enter the number corresponding to your NIM choice: ")) - 1
    if nim_choice not in range(len(nims)):
        print("Invalid choice.")
        return

    selected_model, img_name = nims[nim_choice]
    gpus = input("What GPUs do you want to use? (e.g., '0', '0,1', 'all'): ")
    
    if is_port_in_use(PORT):
        print(f"Port {PORT} is currently in use.")
        kill_process = input("Do you want to kill the process using this port? (y/n): ").strip().lower()
        if kill_process == 'y':
            stop_all_containers()
            time.sleep(5)  # Give some time for the process to be fully killed
            if is_port_in_use(PORT):
                print(f"Port {PORT} is still in use. Cannot start the container.")
                return
        else:
            print("Port is in use. Cannot start the container.")
            return
    else:
        stop_all_containers()

    container_process = start_container(img_name, gpus, os.path.expanduser("~/.cache/nim"))
    if container_process is None:
        print("Test aborted due to missing GPU selection or model identification failure.")
        return

    wait_for_uvicorn(container_process)

    base_url = f"http://0.0.0.0:{PORT}"

    print("Model is alive and ready. Proceeding with the test.")

    print("\n1. Run Manual Test")
    print("2. Auto Test (Scale until TPS < 12)")

    test_choice = input("Enter your choice: ").strip()
    if test_choice == '1':
        total_requests = int(input("Enter the total number of requests to send across all rounds: "))
        concurrency_level = int(input("Enter the number of requests to send simultaneously (concurrency level): "))
        await run_benchmark_tests(base_url, total_requests, concurrency_level, selected_model)
    elif test_choice == '2':
        initial_concurrency = 10  # Start with a medium load
        await auto_scale_benchmark(base_url, initial_concurrency, selected_model, os.environ[NGC_KEY_ENV_VAR])
    else:
        print("Invalid option.")

def manage_nims():
    while True:
        print("\nNIM Management")
        print("1. List NIMs")
        print("2. Add a NIM")
        print("3. Back to Main Menu")
        choice = input("Enter your choice: ").strip()
        if choice == '1':
            list_nims()
        elif choice == '2':
            add_nim()
        elif choice == '3':
            break
        else:
            print("Invalid option.")

def menu():
    ngc_key = load_ngc_key_from_file()

    while True:
        print("Select an option:")
        print("1. Run Manual Test")
        print("2. Auto Test (Scale until TPS < 12)")
        if ngc_key:
            print("3. Change NGC Key")
        else:
            print("3. Input NGC Key")
        print("4. Manage NIMs")
        print("5. Quit")

        choice = input("Enter your choice: ").strip()
        if choice == '1':
            if not os.environ.get(NGC_KEY_ENV_VAR):
                print("NGC API key not found. Please input your NGC API key first.")
            else:
                asyncio.run(run_tests_menu())
        elif choice == '2':
            if not os.environ.get(NGC_KEY_ENV_VAR):
                print("NGC API key not found. Please input your NGC API key first.")
            else:
                asyncio.run(run_tests_menu())
        elif choice == '3':
            change_ngc_key()
            ngc_key = load_ngc_key_from_file()
        elif choice == '4':
            manage_nims()
        elif choice == '5':
            break
        else:
            print("Invalid option.")

if __name__ == "__main__":
    menu()

