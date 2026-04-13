import multiprocessing

def execute_with_timeout(timeout, func, args):
    """Run a function in a child process and return `0` on timeout or worker failure."""
    with multiprocessing.Manager() as manager:
        result_dict = manager.dict()
        
        def worker(result_dict, *args):
            try:
                result = func(*args)
                result_dict['result'] = result
            except Exception as e:
                result_dict['error'] = str(e)
        
        process = multiprocessing.Process(target=worker, args=(result_dict, *args))
        process.start()
        process.join(timeout)
        
        if process.is_alive():
            print("Function exceeded the time limit. Terminating...")
            process.terminate()
            process.join()
            return 0
        
        if 'result' in result_dict:
            return result_dict['result']
        elif 'error' in result_dict:
            print(f"Error in worker: {result_dict['error']}")
            return 0
        else:
            print("No result returned")
            return 0
