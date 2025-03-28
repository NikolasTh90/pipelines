def try_import_from_paths(import_paths, logger=None):
    """
    Try importing a module from multiple possible paths
    
    Args:
        import_paths: List of import statements to try
        logger: Optional logger to use
        
    Returns:
        tuple: (success, imported_module or None)
    """
    for import_path in import_paths:
        try:
            if logger:
                logger.info(f"Trying import: {import_path}")
            
            # Extract the module name (last part after space)
            module_name = import_path.split()[-1]
            
            # Create locals dict to hold the imported module
            locals_dict = {}
            
            # Execute the import
            exec(import_path, globals(), locals_dict)
            
            if logger:
                logger.info(f"{module_name} successfully imported!")
                
            # Return the imported module
            return True, locals_dict[module_name]
        except ImportError as e:
            if logger:
                logger.warning(f"Import failed: {e}")
        except Exception as e:
            if logger:
                logger.warning(f"Unexpected error during import: {e}")
    
    return False, None
