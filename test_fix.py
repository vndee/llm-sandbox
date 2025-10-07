#!/usr/bin/env python3

import tempfile
from llm_sandbox.language_handlers.r_handler import RHandler

def test_ggplot2_plot_detection_injection():
    """Test that the plot detection code doesn't error when ggplot2 is not loaded."""
    
    handler = RHandler()
    
    # Test basic code injection doesn't error
    basic_r_code = """
print("Hello World")
x <- 1:10
y <- x^2
plot(x, y)
"""
    
    try:
        injected_code = handler.inject_plot_detection_code(basic_r_code)
        print("✅ Basic code injection successful")
        print("Injected code length:", len(injected_code))
        
        # Check that the new library/require interceptors are present
        assert ".original_library <- library" in injected_code
        assert ".original_require <- require" in injected_code
        print("✅ Library/require interceptors found")
        
    except Exception as e:
        print(f"❌ Basic code injection failed: {e}")
        return False
    
    # Test code with ggplot2 
    ggplot2_code = """
library(ggplot2)
data(mtcars)
p <- ggplot(mtcars, aes(x = wt, y = mpg)) + geom_point()
print(p)
"""
    
    try:
        injected_code = handler.inject_plot_detection_code(ggplot2_code)
        print("✅ ggplot2 code injection successful")
        
        # Check that the library interceptors handle ggplot2
        assert 'if (package == "ggplot2"' in injected_code
        print("✅ ggplot2 detection logic found")
        
    except Exception as e:
        print(f"❌ ggplot2 code injection failed: {e}")
        return False
    
    print("✅ All tests passed!")
    return True

if __name__ == "__main__":
    success = test_ggplot2_plot_detection_injection()
    exit(0 if success else 1)