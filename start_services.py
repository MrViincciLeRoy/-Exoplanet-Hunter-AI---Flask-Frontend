import subprocess
import os
import sys
from concurrent.futures import ProcessPoolExecutor
import time

def run_backend():
    try:
        # Change to backend directory
        os.chdir("backend")
        print("Changed to backend directory")
        
        # Install backend requirements
        subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
        print("Backend requirements installed")
        
        # Start uvicorn server
        backend_process = subprocess.Popen(
            ["uvicorn", "app:app", "--port", "8000", "--reload"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print("Backend server started on port 8000")
        return backend_process
    except Exception as e:
        print(f"Error in backend startup: {e}")
        return None

def run_frontend():
    try:
        # Change to frontend directory
        os.chdir("frontend")
        print("Changed to frontend directory")
        
        # Install frontend requirements
        subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
        print("Frontend requirements installed")
        
        # Start frontend app
        frontend_process = subprocess.Popen(
            ["python", "app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print("Frontend app started")
        return frontend_process
    except Exception as e:
        print(f"Error in frontend startup: {e}")
        return None

def main():
    # Store the original directory
    original_dir = os.getcwd()
    
    # Run backend and frontend in parallel
    with ProcessPoolExecutor(max_workers=2) as executor:
        backend_future = executor.submit(run_backend)
        frontend_future = executor.submit(run_frontend)
        
        # Wait for processes to complete (they won't, since they're servers)
        backend_process = backend_future.result()
        frontend_process = frontend_future.result()
        
        # Monitor processes
        try:
            while True:
                if backend_process and backend_process.poll() is not None:
                    print("Backend process terminated")
                    break
                if frontend_process and frontend_process.poll() is not None:
                    print("Frontend process terminated")
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down services...")
            if backend_process:
                backend_process.terminate()
            if frontend_process:
                frontend_process.terminate()
            print("Services stopped")
    
    # Return to original directory
    os.chdir(original_dir)

if __name__ == "__main__":
    main()