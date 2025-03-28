import os
import sys
import time
import traceback
from typing import Optional, List
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Language_Filter")

class Pipeline:
    class Valves(BaseModel):
        # List target pipeline ids (models) that this filter will be connected to
        pipelines: List[str] = ["*"]
        
        # Assign a priority level to the filter pipeline
        priority: int = 0
        
        # Custom configuration for language detection and translation
        expected_languages_iso: List[str] = ["en", "el"]  # Default expected languages: English and Greek
        default_language: str = "en"  # Default language to translate to if all validations fail
        threshold: float = 0.75  # Confidence threshold for language detection
        log_level: str = "info"
        enforce_language: bool = True  # If True, enforces translation when language differs
        use_gpu: bool = False  # Set to False by default to avoid CUDA memory issues
        
    def __init__(self):
        # This filter uses Guardrails AI's CorrectLanguage module to detect and translate content
        self.type = "filter"
        self.name = "Guardrails Language Filter"

        # Initialize with default configuration
        self.valves = self.Valves()
        self.default_language = self.valves.default_language
        
        # Set device for GPU/CPU - default to CPU to avoid memory issues
        self._set_device(self.valves.use_gpu)
        
        # Print diagnostic information
        self._print_environment_info()
        
        # Set logging level based on configuration
        if hasattr(logging, self.valves.log_level.upper()):
            logger.setLevel(getattr(logging, self.valves.log_level.upper()))
        
        # Try to import Guardrails components
        logger.info("Initializing Guardrails Language Filter...")
        self.guardrails_mode = self._setup_guardrails()

    def _set_device(self, use_gpu):
        """Set the device for translation models with fallback handling"""
        if use_gpu:
            try:
                # Check if CUDA is available
                import torch
                if torch.cuda.is_available():
                    os.environ["DEVICE"] = "cuda"
                    logger.info("Set language validator to use GPU (CUDA)")
                    # Check memory to warn about potential issues
                    free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                    free_memory_gb = free_memory / (1024**3)
                    if free_memory_gb < 2.0:  # Less than 2GB free
                        logger.warning(f"Low GPU memory detected ({free_memory_gb:.2f} GB free). This may cause issues.")
                        logger.warning("Consider setting use_gpu=False if the filter fails to initialize.")
                else:
                    logger.warning("CUDA not available. Falling back to CPU for language processing.")
                    os.environ["DEVICE"] = "cpu"
                    self.valves.use_gpu = False
            except Exception as e:
                logger.warning(f"Error checking GPU: {e}. Falling back to CPU for language processing.")
                os.environ["DEVICE"] = "cpu"
                self.valves.use_gpu = False
        else:
            os.environ["DEVICE"] = "cpu" 
            logger.info("Set language validator to use CPU")

    def _print_environment_info(self):
        """Print diagnostic information about the environment"""
        logger.info("=== Environment Information ===")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Python executable: {sys.executable}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
        logger.info(f"DEVICE: {os.environ.get('DEVICE', 'Not set')}")
        
        # Check GPU availability if using GPU
        if self.valves.use_gpu:
            try:
                import torch
                logger.info(f"CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
                    logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
                    # Print memory information
                    free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                    logger.info(f"GPU free memory: {free_memory / (1024**3):.2f} GB")
            except ImportError:
                logger.warning("Torch not available - cannot check CUDA status")
        
        logger.info("===============================")

    def _setup_guardrails(self):
        """Set up Guardrails and return the mode (guardrails or disabled)"""
        setup_start_time = time.time()
        logger.info("Setting up Guardrails language validator...")
        
        # Try to import necessary components - don't attempt installation
        try:
            # Basic guardrails imports
            import guardrails
            from guardrails import Guard
            from guardrails import hub
            
            logger.info(f"Guardrails and its components found")
        except ImportError as e:
            logger.error(f"Required Guardrails components not available: {e}")
            logger.error("Language filter will be disabled")
            setup_end_time = time.time()
            logger.info(f"Setup completed in {setup_end_time - setup_start_time:.2f} seconds (disabled)")
            return "disabled"
        
        # Try to import CorrectLanguage specifically - try different possible locations
        correct_language_module_found = False
        for import_path in [
            "from guardrails.hub import CorrectLanguage",
            "from guardrails_hub_correct_language import CorrectLanguage",
            "from guardrails_grhub_correct_language import CorrectLanguage"
        ]:
            try:
                logger.info(f"Trying import: {import_path}")
                exec(import_path, globals())
                logger.info("CorrectLanguage successfully imported!")
                correct_language_module_found = True
                break
            except ImportError as e:
                logger.warning(f"Import failed: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error during import: {e}")
        
        if not correct_language_module_found:
            logger.error("CorrectLanguage could not be imported from any known location")
            logger.error("Language filter will be disabled")
            setup_end_time = time.time()
            logger.info(f"Setup completed in {setup_end_time - setup_start_time:.2f} seconds (disabled)")
            return "disabled"
        
        # Try to instantiate CorrectLanguage and create Guard
        try:
            # Test instantiation with error handling
            logger.info(f"Attempting to instantiate CorrectLanguage...")
            
            # Test the validator
            from guardrails import Guard
            
            try:
                test_guard = Guard().use(
                    CorrectLanguage,
                    expected_language_iso=self.valves.default_language,
                    threshold=self.valves.threshold,
                    on_fail="noop"
                )
                logger.info("Guard successfully created with CorrectLanguage")
            except RuntimeError as e:
                # Check if this is a CUDA memory error
                error_str = str(e)
                if "CUDA out of memory" in error_str or "Failed to set up the translation pipeline" in error_str:
                    logger.warning("GPU memory error detected when initializing language validator")
                    logger.warning("Falling back to CPU for language validation...")
                    
                    # Switch to CPU and try again
                    self.valves.use_gpu = False
                    self._set_device(False)
                    
                    # Retry with CPU
                    test_guard = Guard().use(
                        CorrectLanguage,
                        expected_language_iso=self.valves.default_language,
                        threshold=self.valves.threshold,
                        on_fail="noop"
                    )
                    logger.info("Guard successfully created with CorrectLanguage on CPU")
                else:
                    # Re-raise the error if it's not a CUDA memory issue
                    raise
            
            # Test validation with a simple string - just to verify it's working
            try:
                logger.info("Testing validator with a simple English string...")
                validation_start = time.time()
                result = test_guard.validate("This is a test string.")
                validation_end = time.time()
                
                # Log the validation result
                logger.info(f"Validation outcome: validation_passed={result.validation_passed}")
                logger.info(f"Validation test took {validation_end - validation_start:.2f} seconds")
                
                # Log if the device is actually being used
                logger.info(f"Device being used: {os.environ.get('DEVICE', 'Not reported')}")
            except Exception as e:
                logger.warning(f"Validation test failed: {e}")
                logger.warning("This may indicate issues with the validator, but we'll continue")
            
            setup_end_time = time.time()
            logger.info(f"Setup completed in {setup_end_time - setup_start_time:.2f} seconds (guardrails mode)")
            return "guardrails"
        except Exception as e:
            logger.error(f"Error setting up CorrectLanguage or Guard: {e}")
            logger.error("Full traceback:")
            traceback.print_exc()
            setup_end_time = time.time()
            logger.info(f"Setup completed in {setup_end_time - setup_start_time:.2f} seconds (disabled)")
            return "disabled"

    async def on_startup(self):
        # This function is called when the server is started
        logger.info(f"Guardrails Language Filter starting up")
        
        if self.guardrails_mode == "guardrails":
            logger.info(f"Running with Guardrails CorrectLanguage validator")
            logger.info(f"Expected languages: {self.valves.expected_languages_iso}")
            logger.info(f"Default language: {self.valves.default_language}")
            logger.info(f"Detection threshold: {self.valves.threshold}")
            logger.info(f"Using {'GPU' if self.valves.use_gpu else 'CPU'} for detection and translation")
        else:
            logger.error("Language filter is disabled - CorrectLanguage validator could not be loaded")
            logger.info("The filter will pass messages through unmodified")
            
        logger.info(f"Enforce language: {self.valves.enforce_language}")

    async def on_shutdown(self):
        # This function is called when the server is stopped
        logger.info("Guardrails Language Filter shutting down")

    async def on_valves_updated(self):
        # This function is called when the valves are updated
        logger.info(f"Valves updated: Expected languages = {self.valves.expected_languages_iso}")
        logger.info(f"Default language = {self.valves.default_language}")
        logger.info(f"Detection threshold: {self.valves.threshold}")
        logger.info(f"Enforce language: {self.valves.enforce_language}")
        logger.info(f"Using GPU: {self.valves.use_gpu}")
        
        # Update default language
        self.default_language = self.valves.default_language
        
        # Update GPU/CPU setting if it changed
        self._set_device(self.valves.use_gpu)
        
        # Update the logging level if it changed
        if hasattr(logging, self.valves.log_level.upper()):
            logger.setLevel(getattr(logging, self.valves.log_level.upper()))
            logger.info(f"Updated log level to {self.valves.log_level.upper()}")

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Pass through inlet - no processing needed
        """
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Process outgoing content according to the specified workflow:
        1. Check if content passes validation for any of the expected languages concurrently
        2. If not, translate it to the default language
        3. Ensure proper GPU memory cleanup throughout
        """
        # Safely check if content is already blocked
        if body.get("_blocked"):
            if body.get("_raise_block"):
                raise Exception(body.get("_blocked_reason", "Content blocked"))
            else:
                return body
        outlet_start = time.time()
        
        # If Guardrails not available, just pass through
        if self.guardrails_mode != "guardrails":
            logger.warning("Language filter is disabled - passing through unmodified")
            return body
        
        # Extract the content to validate
        if "messages" in body and body["messages"]:
            content_to_validate = body["messages"][-1].get("content")
        else:
            logger.warning("Could not find content to validate")
            return body
        
        logger.info(f"Validating content: {content_to_validate[:50]}...")
        
        try:
            import concurrent.futures
            from guardrails import Guard
            
            def validate_language(language):
                """Validates content against a specific language and returns (language, is_valid)"""
                try:
                    logger.info(f"Testing if content is in language: {language}")
                    # Create a guard with noop on_fail - we just want to check if it passes
                    temp_guard = Guard().use(
                        CorrectLanguage,
                        expected_language_iso=language,
                        threshold=self.valves.threshold,
                        on_fail="noop"  # Don't modify content, just check
                    )
                    
                    # Validate against this language
                    validation_result = temp_guard.validate(content_to_validate)
                    is_valid = validation_result.validation_passed
                    
                    logger.info(f"Validation for {language}: {is_valid}")
                    return language, is_valid
                except Exception as e:
                    logger.error(f"Error validating for {language}: {e}")
                    return language, False
                finally:
                    # Free GPU memory if using GPU
                    if self.valves.use_gpu:
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                logger.debug(f"Cleared CUDA cache after {language} validation")
                        except Exception as e:
                            logger.warning(f"Failed to clear GPU memory: {e}")
            
            # Use a thread pool to run validations concurrently
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit all validation tasks to the executor
                future_to_language = {
                    executor.submit(validate_language, lang): lang 
                    for lang in self.valves.expected_languages_iso
                }
                
                # Check results as they come in, for fast short-circuiting
                for future in concurrent.futures.as_completed(future_to_language):
                    language, is_valid = future.result()
                    if is_valid:
                        logger.info(f"Content passed validation for {language} - returning unmodified")
                        # Cancel all remaining tasks to save resources
                        for f in future_to_language:
                            if not f.done() and f != future:
                                f.cancel()
                        
                        # Clear GPU memory before returning
                        self._clear_gpu_memory()
                        return body
            
            # If we get here, content failed validation for all expected languages
            logger.info("Content failed validation for all expected languages - translating to default language")
            
            # Only translate if enforce_language is True
            if not self.valves.enforce_language:
                logger.info("Language enforcement is disabled - passing through unmodified")
                self._clear_gpu_memory()
                return body
                
            try:
                # Create a new guard to translate to default language
                translator_guard = Guard().use(
                    CorrectLanguage,
                    expected_language_iso=self.default_language,
                    threshold=self.valves.threshold,
                    on_fail="fix"  # This will translate the content
                )
                
                # Translate the content
                translation_result = translator_guard.validate(content_to_validate)
                
                # Check if we have a translation
                if hasattr(translation_result, 'validated_output') and translation_result.validated_output:
                    translated_content = translation_result.validated_output
                    logger.info(f"Translated to {self.default_language}: {translated_content[:50]}...")
                    
                    # Update the content in the body
                    body["messages"][-1]["content"] = translated_content
                else:
                    logger.warning("Translation failed - no validated output available")
            except Exception as e:
                logger.error(f"Error during translation: {e}")
            finally:
                # Ensure GPU memory is cleared after translation
                self._clear_gpu_memory()
                
        except Exception as e:
            logger.error(f"Error during language validation/translation: {e}")
            logger.error(traceback.format_exc())
            # Ensure GPU memory is cleared even if an exception occurs
            self._clear_gpu_memory()
        
        outlet_end = time.time()
        logger.info(f"Outlet processing completed in {outlet_end - outlet_start:.2f} seconds")
        self._clear_gpu_memory()
        
        return body

    def _clear_gpu_memory(self):
        """Helper method to clear GPU memory"""
        if self.valves.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    # Free cache
                    torch.cuda.empty_cache()
                    logger.debug("Cleared CUDA cache")
                    
                    # Report memory stats for debugging
                    if logger.level <= logging.DEBUG:
                        allocated = torch.cuda.memory_allocated() / (1024**3)
                        reserved = torch.cuda.memory_reserved() / (1024**3)
                        logger.debug(f"GPU memory after cleanup: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")
            except Exception as e:
                logger.warning(f"Failed to clear GPU memory: {e}")