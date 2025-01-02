import threading
import time

# Event and progress for the first thread
pause_event_1 = threading.Event()
pause_event_1.set()  # Initially set to allow execution
progress_1 = 0
total_work = 100

# Event and progress for the second thread
pause_event_2 = threading.Event()
pause_event_2.set()  # Initially set to allow execution
progress_2 = 0

# Lock for thread-safe progress updates
progress_lock_1 = threading.Lock()
progress_lock_2 = threading.Lock()

# Function to be executed by each thread


def log_elapsed_time(thread_num):
    global progress_1, progress_2
    while True:
        if thread_num == 1:
            print(f"Thread {thread_num} is waiting for pause_event_1...")
            pause_event_1.wait()  # Pause if event is cleared
            # Simulate work
            print(f"Thread {thread_num} is executing work...")
            time.sleep(0.1)
            with progress_lock_1:  # Ensure thread-safe updates
                progress_1 += 1
                print(f"Thread {thread_num} Progress updated: {
                      progress_1}/{total_work} ({(progress_1 / total_work) * 100:.2f}%)")
            if progress_1 >= total_work:
                break
        elif thread_num == 2:
            print(f"Thread {thread_num} is waiting for pause_event_2...")
            pause_event_2.wait()  # Pause if event is cleared
            # Simulate work
            print(f"Thread {thread_num} is executing work...")
            time.sleep(0.1)
            with progress_lock_2:  # Ensure thread-safe updates
                progress_2 += 1
                print(f"Thread {thread_num} Progress updated: {
                      progress_2}/{total_work} ({(progress_2 / total_work) * 100:.2f}%)")
            if progress_2 >= total_work:
                break

    print(f"Thread {thread_num} Work complete!")


# Start the threads
thread_1 = threading.Thread(target=log_elapsed_time, args=(1,), daemon=True)
thread_2 = threading.Thread(target=log_elapsed_time, args=(2,), daemon=True)

thread_1.start()
thread_2.start()

# Function to pause the threads


def pause_thread(thread_num):
    if thread_num == 1:
        print("Pausing thread 1...")
        pause_event_1.clear()
    elif thread_num == 2:
        print("Pausing thread 2...")
        pause_event_2.clear()

# Function to resume the threads


def resume_thread(thread_num):
    if thread_num == 1:
        print("Resuming thread 1...")
        pause_event_1.set()
    elif thread_num == 2:
        print("Resuming thread 2...")
        pause_event_2.set()


# Example usage:
time.sleep(3)
pause_thread(1)  # Pauses thread 1 after 3 seconds
pause_thread(2)  # Pauses thread 2 after 3 seconds
time.sleep(2)
resume_thread(1)  # Resumes thread 1 after 2 seconds
resume_thread(2)  # Resumes thread 2 after 2 seconds
