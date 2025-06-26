# ruff: noqa: SLF001, PLR2004

import logging
import re
from unittest.mock import MagicMock

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.language_handlers.java_handler import JavaHandler


class TestJavaHandler:
    """Test JavaHandler specific functionality."""

    def test_init(self) -> None:
        """Test JavaHandler initialization."""
        handler = JavaHandler()

        assert handler.config.name == SupportedLanguage.JAVA
        assert handler.config.file_extension == "java"
        assert "java {file}" in handler.config.execution_commands
        assert handler.config.package_manager == "mvn"
        assert handler.config.plot_detection is None
        assert handler.config.is_support_library_installation is False

    def test_init_with_custom_logger(self) -> None:
        """Test JavaHandler initialization with custom logger."""
        custom_logger = logging.getLogger("custom")
        handler = JavaHandler(custom_logger)
        assert handler.logger == custom_logger

    def test_inject_plot_detection_code(self) -> None:
        """Test plot detection code injection (should return unchanged code)."""
        handler = JavaHandler()
        code = 'System.out.println("Hello World");'

        injected_code = handler.inject_plot_detection_code(code)

        assert injected_code == code  # Should be unchanged since Java doesn't support plot detection

    def test_run_with_artifacts_no_plotting_support(self) -> None:
        """Test run_with_artifacts returns empty plots list."""
        handler = JavaHandler()
        mock_container = MagicMock()
        mock_result = MagicMock()
        mock_container.run.return_value = mock_result

        result, plots = handler.run_with_artifacts(
            container=mock_container,
            code='System.out.println("Hello");',
            libraries=None,
            enable_plotting=True,
            timeout=30,
            output_dir="/tmp/sandbox_plots",
        )

        assert result == mock_result
        assert plots == []
        mock_container.run.assert_called_once_with('System.out.println("Hello");', None, timeout=30)

    def test_extract_plots_returns_empty(self) -> None:
        """Test extract_plots returns empty list."""
        handler = JavaHandler()
        mock_container = MagicMock()

        plots = handler.extract_plots(mock_container, "/tmp/sandbox_plots")

        assert plots == []

    def test_get_import_patterns(self) -> None:
        """Test get_import_patterns method."""
        handler = JavaHandler()

        # Test basic package import
        pattern = handler.get_import_patterns("java.util")

        # Should match various import formats
        import_code_samples = [
            "import java.util.List;",
            "import java.util.*;",
            "import java.util.ArrayList;",
            "  import java.util.HashMap;  ",
        ]

        for code in import_code_samples:
            assert re.search(pattern, code), f"Pattern should match: {code}"

        # Test specific class import
        list_pattern = handler.get_import_patterns("java.util.List")
        assert re.search(list_pattern, "import java.util.List;")
        assert not re.search(list_pattern, "import java.util.ArrayList;")

        # Should not match comments or parts of other words
        non_matching_samples = [
            "// import java.util.List;",
            "/* import java.util.*; */",
            "import java.utility.List;",
            "import com.java.util.List;",
        ]

        for code in non_matching_samples:
            filtered_code = handler.filter_comments(code)
            assert not re.search(pattern, filtered_code), f"Pattern should not match: {code}"

    def test_get_multiline_comment_patterns(self) -> None:
        """Test get_multiline_comment_patterns method."""
        pattern = JavaHandler.get_multiline_comment_patterns()

        comment_samples = [
            "/* This is a comment */",
            "/*\n * Multiline\n * comment\n */",
            "/* Single line with content */",
            "/**\n * Javadoc comment\n */",
        ]

        for comment in comment_samples:
            assert re.search(pattern, comment), f"Pattern should match: {comment}"

    def test_get_inline_comment_patterns(self) -> None:
        """Test get_inline_comment_patterns method."""
        pattern = JavaHandler.get_inline_comment_patterns()

        comment_samples = [
            "// This is a comment",
            'System.out.println("Hello");  // Inline comment',
            "    // Indented comment",
            "int x = 5; // Variable definition",
        ]

        for comment in comment_samples:
            assert re.search(pattern, comment), f"Pattern should match: {comment}"

    def test_filter_comments(self) -> None:
        """Test comment filtering functionality."""
        handler = JavaHandler()

        code_with_comments = """
        // This is a single line comment
        public class Test {
            /* This is a
               multiline comment */
            public static void main(String[] args) {
                System.out.println("Hello"); // Inline comment
            }
        }
        """

        filtered_code = handler.filter_comments(code_with_comments)

        # Should remove comments but keep code
        assert 'System.out.println("Hello");' in filtered_code
        assert "public class Test" in filtered_code
        assert "// This is a single line comment" not in filtered_code
        assert "/* This is a" not in filtered_code
        assert "// Inline comment" not in filtered_code

    def test_properties(self) -> None:
        """Test handler property methods."""
        handler = JavaHandler()

        assert handler.name == SupportedLanguage.JAVA
        assert handler.file_extension == "java"
        assert handler.supported_plot_libraries == []
        assert handler.is_support_library_installation is False
        assert handler.is_support_plot_detection is False

    def test_get_execution_commands(self) -> None:
        """Test getting execution commands."""
        handler = JavaHandler()

        commands = handler.get_execution_commands("Test.java")

        assert len(commands) == 1
        assert commands[0] == "java Test.java"

    def test_get_library_installation_command(self) -> None:
        """Test getting library installation command."""
        handler = JavaHandler()

        command = handler.get_library_installation_command("junit")

        assert command == "mvn junit"

    def test_complex_import_patterns(self) -> None:
        """Test more complex import scenarios."""
        handler = JavaHandler()

        # Test nested package imports
        pattern = handler.get_import_patterns("com.example.util")

        complex_imports = [
            "import com.example.util.Helper;",
            "import com.example.util.*;",
            "import com.example.util.data.Model;",
        ]

        for import_stmt in complex_imports:
            if "com.example.util.data" not in import_stmt:  # This should not match the pattern
                assert re.search(pattern, import_stmt), f"Should match: {import_stmt}"
