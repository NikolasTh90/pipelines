import concurrent.futures

def run_tasks_concurrently(tasks, cancel_on_first_success=False):
    """
    Run multiple tasks concurrently and return their results
    
    Args:
        tasks: List of (function, args) tuples to run
        cancel_on_first_success: Whether to cancel remaining tasks when first success is found
        
    Returns:
        list: Results from all completed tasks
    """
    results = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all tasks to the executor
        future_to_task = {
            executor.submit(func, *args): (func, args) 
            for func, args in tasks
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_task):
            result = future.result()
            results.append(result)
            
            # Cancel remaining tasks if requested and this task succeeded
            if cancel_on_first_success:
                success = False
                if isinstance(result, tuple) and len(result) >= 2:
                    success = result[1]
                elif isinstance(result, bool):
                    success = result
                
                if success:
                    print("Task succeeded, canceling remaining tasks")
                    for f in future_to_task:
                        if not f.done() and f != future:
                            f.cancel()
                    break
    
    return results
