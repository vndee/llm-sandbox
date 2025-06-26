# ruff: noqa: SLF001, PLR2004

import logging
import re
from unittest.mock import MagicMock

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.language_handlers.cpp_handler import CppHandler


class TestCppHandler:
    """Test CppHandler specific functionality."""

    def test_init(self) -> None:
        """Test CppHandler initialization."""
        handler = CppHandler()

        assert handler.config.name == SupportedLanguage.CPP
        assert handler.config.file_extension == "cpp"
        assert "g++ -std=c++17 {file} -o /tmp/a.out && /tmp/a.out" in handler.config.execution_commands
        assert handler.config.package_manager == "apt-get install"
        assert handler.config.plot_detection is None
        assert handler.config.is_support_library_installation is True

    def test_init_with_custom_logger(self) -> None:
        """Test CppHandler initialization with custom logger."""
        custom_logger = logging.getLogger("custom")
        handler = CppHandler(custom_logger)
        assert handler.logger == custom_logger

    def test_inject_plot_detection_code(self) -> None:
        """Test plot detection code injection (should return unchanged code)."""
        handler = CppHandler()
        code = '#include <iostream>\nint main() { std::cout << "Hello"; return 0; }'

        injected_code = handler.inject_plot_detection_code(code)

        assert injected_code == code  # Should be unchanged since C++ doesn't support plot detection

    def test_run_with_artifacts_no_plotting_support(self) -> None:
        """Test run_with_artifacts returns empty plots list."""
        handler = CppHandler()
        mock_container = MagicMock()
        mock_result = MagicMock()
        mock_container.run.return_value = mock_result

        result, plots = handler.run_with_artifacts(
            container=mock_container,
            code="#include <iostream>\nint main() { return 0; }",
            libraries=["libstdc++"],
            enable_plotting=True,
            timeout=30,
            output_dir="/tmp/sandbox_plots",
        )

        assert result == mock_result
        assert plots == []
        mock_container.run.assert_called_once_with(
            "#include <iostream>\nint main() { return 0; }", ["libstdc++"], timeout=30
        )

    def test_extract_plots_returns_empty(self) -> None:
        """Test extract_plots returns empty list."""
        handler = CppHandler()
        mock_container = MagicMock()

        plots = handler.extract_plots(mock_container, "/tmp/sandbox_plots")

        assert plots == []

    def test_get_import_patterns_system_headers(self) -> None:
        """Test get_import_patterns method for system headers."""
        handler = CppHandler()

        pattern = handler.get_import_patterns("iostream")

        # Should match system header includes
        system_includes = [
            "#include <iostream>",
            "#include  <iostream>",
            "  #include <iostream>",
            "#include<iostream>",
        ]

        for code in system_includes:
            assert re.search(pattern, code), f"Pattern should match system header: {code}"

    def test_get_import_patterns_local_headers(self) -> None:
        """Test get_import_patterns method for local headers."""
        handler = CppHandler()

        pattern = handler.get_import_patterns("myheader.h")

        # Should match local header includes
        local_includes = [
            '#include "myheader.h"',
            '#include  "myheader.h"',
            '  #include "myheader.h"',
            '#include"myheader.h"',
        ]

        for code in local_includes:
            assert re.search(pattern, code), f"Pattern should match local header: {code}"

    def test_get_import_patterns_no_false_positives(self) -> None:
        """Test that import patterns don't match unrelated code."""
        handler = CppHandler()

        pattern = handler.get_import_patterns("vector")

        # Should not match comments or parts of other words
        non_matching_samples = [
            "// #include <vector>",
            "/* #include <vector> */",
            "#include <vectors>",  # Different header
            "std::vector<int> v;",  # Usage, not include
            'std::cout << "#include <vector>";',  # String literal
        ]

        for code in non_matching_samples:
            filtered_code = handler.filter_comments(code)
            if "#include <vectors>" not in code and "std::" not in code:
                assert not re.search(pattern, filtered_code), f"Pattern should not match: {code}"

    def test_get_multiline_comment_patterns(self) -> None:
        """Test get_multiline_comment_patterns method."""
        pattern = CppHandler.get_multiline_comment_patterns()

        comment_samples = [
            "/* This is a comment */",
            "/*\n * Multiline\n * comment\n */",
            "/* Single line with content */",
            "/**\n * Documentation comment\n */",
        ]

        for comment in comment_samples:
            assert re.search(pattern, comment), f"Pattern should match: {comment}"

    def test_get_inline_comment_patterns(self) -> None:
        """Test get_inline_comment_patterns method."""
        pattern = CppHandler.get_inline_comment_patterns()

        comment_samples = [
            "// This is a comment",
            "int x = 5;  // Inline comment",
            "    // Indented comment",
            'std::cout << "Hello"; // Output statement',
        ]

        for comment in comment_samples:
            assert re.search(pattern, comment), f"Pattern should match: {comment}"

    def test_filter_comments(self) -> None:
        """Test comment filtering functionality."""
        handler = CppHandler()

        code_with_comments = """
        // This is a single line comment
        #include <iostream>
        int main() {
            /* This is a
               multiline comment */
            std::cout << "Hello"; // Inline comment
            return 0;
        }
        """

        filtered_code = handler.filter_comments(code_with_comments)

        # Should remove comments but keep code
        assert 'std::cout << "Hello";' in filtered_code
        assert "#include <iostream>" in filtered_code
        assert "int main()" in filtered_code
        assert "// This is a single line comment" not in filtered_code
        assert "/* This is a" not in filtered_code
        assert "// Inline comment" not in filtered_code

    def test_properties(self) -> None:
        """Test handler property methods."""
        handler = CppHandler()

        assert handler.name == SupportedLanguage.CPP
        assert handler.file_extension == "cpp"
        assert handler.supported_plot_libraries == []
        assert handler.is_support_library_installation is True
        assert handler.is_support_plot_detection is False

    def test_get_execution_commands(self) -> None:
        """Test getting execution commands."""
        handler = CppHandler()

        commands = handler.get_execution_commands("main.cpp")

        assert len(commands) == 2
        assert commands[0] == "g++ -o a.out main.cpp"
        assert commands[1] == "./a.out"

    def test_get_library_installation_command(self) -> None:
        """Test getting library installation command."""
        handler = CppHandler()

        command = handler.get_library_installation_command("libboost-dev")

        assert command == "apt-get install libboost-dev"

    def test_standard_library_headers(self) -> None:
        """Test standard library header imports."""
        handler = CppHandler()

        standard_headers = [
            "iostream",
            "vector",
            "string",
            "algorithm",
            "map",
            "set",
            "queue",
            "stack",
            "memory",
            "thread",
        ]

        for header in standard_headers:
            pattern = handler.get_import_patterns(header)
            include_stmt = f"#include <{header}>"
            assert re.search(pattern, include_stmt), f"Should match standard header: {include_stmt}"

    def test_header_with_extension(self) -> None:
        """Test headers with file extensions."""
        handler = CppHandler()

        headers_with_ext = [
            ("stdio.h", "#include <stdio.h>"),
            ("stdlib.h", "#include <stdlib.h>"),
            ("math.h", "#include <math.h>"),
            ("string.h", "#include <string.h>"),
        ]

        for header, include_stmt in headers_with_ext:
            pattern = handler.get_import_patterns(header)
            assert re.search(pattern, include_stmt), f"Should match C header: {include_stmt}"

    def test_custom_headers(self) -> None:
        """Test custom/local header includes."""
        handler = CppHandler()

        custom_headers = [
            ("myclass.h", '#include "myclass.h"'),
            ("utils.hpp", '#include "utils.hpp"'),
            ("../include/helper.h", '#include "../include/helper.h"'),
            ("config.h", '#include "config.h"'),
        ]

        for header, include_stmt in custom_headers:
            pattern = handler.get_import_patterns(header)
            assert re.search(pattern, include_stmt), f"Should match custom header: {include_stmt}"

    def test_complex_include_scenarios(self) -> None:
        """Test complex include scenarios."""
        handler = CppHandler()

        # Test that it doesn't match partial names
        vector_pattern = handler.get_import_patterns("vector")

        # Should match
        assert re.search(vector_pattern, "#include <vector>")

        # Should NOT match these
        assert not re.search(vector_pattern, "#include <vectormath>")
        assert not re.search(vector_pattern, "#include <myvector>")

    def test_preprocessor_directives_not_matched_as_includes(self) -> None:
        """Test that other preprocessor directives are not matched as includes."""
        handler = CppHandler()

        pattern = handler.get_import_patterns("NDEBUG")

        # These should NOT match the include pattern
        preprocessor_directives = [
            "#define NDEBUG",
            "#ifndef NDEBUG",
            "#ifdef NDEBUG",
            "#undef NDEBUG",
        ]

        for directive in preprocessor_directives:
            assert not re.search(pattern, directive), f"Should not match preprocessor directive: {directive}"
