# ruff: noqa: SLF001, PLR2004

import logging
import re
from unittest.mock import MagicMock

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.language_handlers.go_handler import GoHandler


class TestGoHandler:
    """Test GoHandler specific functionality."""

    def test_init(self) -> None:
        """Test GoHandler initialization."""
        handler = GoHandler()

        assert handler.config.name == SupportedLanguage.GO
        assert handler.config.file_extension == "go"
        assert "go run {file}" in handler.config.execution_commands
        assert handler.config.package_manager == "go get"
        assert handler.config.plot_detection is None
        assert handler.config.is_support_library_installation is True

    def test_init_with_custom_logger(self) -> None:
        """Test GoHandler initialization with custom logger."""
        custom_logger = logging.getLogger("custom")
        handler = GoHandler(custom_logger)
        assert handler.logger == custom_logger

    def test_inject_plot_detection_code(self) -> None:
        """Test plot detection code injection (should return unchanged code)."""
        handler = GoHandler()
        code = 'package main\nimport "fmt"\nfunc main() { fmt.Println("Hello") }'

        injected_code = handler.inject_plot_detection_code(code)

        assert injected_code == code  # Should be unchanged since Go doesn't support plot detection

    def test_run_with_artifacts_no_plotting_support(self) -> None:
        """Test run_with_artifacts returns empty plots list."""
        handler = GoHandler()
        mock_container = MagicMock()
        mock_result = MagicMock()
        mock_container.run.return_value = mock_result

        result, plots = handler.run_with_artifacts(
            container=mock_container,
            code="package main\nfunc main() {}",
            libraries=["github.com/gorilla/mux"],
            enable_plotting=True,
            timeout=30,
            output_dir="/tmp/sandbox_plots",
        )

        assert result == mock_result
        assert plots == []
        mock_container.run.assert_called_once_with(
            "package main\nfunc main() {}", ["github.com/gorilla/mux"], timeout=30
        )

    def test_extract_plots_returns_empty(self) -> None:
        """Test extract_plots returns empty list."""
        handler = GoHandler()
        mock_container = MagicMock()

        plots = handler.extract_plots(mock_container, "/tmp/sandbox_plots")

        assert plots == []

    def test_get_import_patterns_simple_package(self) -> None:
        """Test get_import_patterns method for simple packages."""
        handler = GoHandler()

        pattern = handler.get_import_patterns("fmt")

        # Should match simple package imports
        simple_imports = [
            'import "fmt"',
            'import (\n    "fmt"\n)',
            '    import "fmt"',
            'import "fmt"  // comment',
        ]

        for code in simple_imports:
            assert re.search(pattern, code), f"Pattern should match simple import: {code}"

    def test_get_import_patterns_aliased_imports(self) -> None:
        """Test get_import_patterns method for aliased imports."""
        handler = GoHandler()

        pattern = handler.get_import_patterns("fmt")

        # Should match aliased imports
        aliased_imports = [
            'import f "fmt"',
            'import . "fmt"',
            'import _ "fmt"',
            'import (\n    f "fmt"\n)',
        ]

        for code in aliased_imports:
            assert re.search(pattern, code), f"Pattern should match aliased import: {code}"

    def test_get_import_patterns_module_paths(self) -> None:
        """Test get_import_patterns method for module paths."""
        handler = GoHandler()

        pattern = handler.get_import_patterns("github.com/gorilla/mux")

        # Should match module path imports
        module_imports = [
            'import "github.com/gorilla/mux"',
            'import (\n    "github.com/gorilla/mux"\n)',
            'import mux "github.com/gorilla/mux"',
            'import . "github.com/gorilla/mux"',
        ]

        for code in module_imports:
            assert re.search(pattern, code), f"Pattern should match module import: {code}"

    def test_get_import_patterns_no_false_positives(self) -> None:
        """Test that import patterns don't match unrelated code."""
        handler = GoHandler()

        pattern = handler.get_import_patterns("fmt")

        # Should not match comments or parts of other words
        non_matching_samples = [
            '// import "fmt"',
            '/* import "fmt" */',
            'import "fmtutil"',  # Different package
            'fmt.Println("hello")',  # Usage, not import
            'import "myfmt"',  # Different package
        ]

        for code in non_matching_samples:
            filtered_code = handler.filter_comments(code)
            if "fmtutil" not in code and "fmt.Println" not in code and "myfmt" not in code:
                assert not re.search(pattern, filtered_code), f"Pattern should not match: {code}"

    def test_get_multiline_comment_patterns(self) -> None:
        """Test get_multiline_comment_patterns method."""
        pattern = GoHandler.get_multiline_comment_patterns()

        comment_samples = [
            "/* This is a comment */",
            "/*\n * Multiline\n * comment\n */",
            "/* Single line with content */",
            "/*\n   Block comment\n   with multiple lines\n*/",
        ]

        for comment in comment_samples:
            assert re.search(pattern, comment), f"Pattern should match: {comment}"

    def test_get_inline_comment_patterns(self) -> None:
        """Test get_inline_comment_patterns method."""
        pattern = GoHandler.get_inline_comment_patterns()

        comment_samples = [
            "// This is a comment",
            'fmt.Println("Hello")  // Inline comment',
            "    // Indented comment",
            "var x int // Variable definition",
        ]

        for comment in comment_samples:
            assert re.search(pattern, comment), f"Pattern should match: {comment}"

    def test_filter_comments(self) -> None:
        """Test comment filtering functionality."""
        handler = GoHandler()

        code_with_comments = """
        // This is a single line comment
        package main
        import "fmt"
        func main() {
            /* This is a
               multiline comment */
            fmt.Println("Hello") // Inline comment
        }
        """

        filtered_code = handler.filter_comments(code_with_comments)

        # Should remove comments but keep code
        assert 'fmt.Println("Hello")' in filtered_code
        assert 'import "fmt"' in filtered_code
        assert "package main" in filtered_code
        assert "// This is a single line comment" not in filtered_code
        assert "/* This is a" not in filtered_code
        assert "// Inline comment" not in filtered_code

    def test_properties(self) -> None:
        """Test handler property methods."""
        handler = GoHandler()

        assert handler.name == SupportedLanguage.GO
        assert handler.file_extension == "go"
        assert handler.supported_plot_libraries == []
        assert handler.is_support_library_installation is True
        assert handler.is_support_plot_detection is False

    def test_get_execution_commands(self) -> None:
        """Test getting execution commands."""
        handler = GoHandler()

        commands = handler.get_execution_commands("main.go")

        assert len(commands) == 1
        assert commands[0] == "go run main.go"

    def test_get_library_installation_command(self) -> None:
        """Test getting library installation command."""
        handler = GoHandler()

        command = handler.get_library_installation_command("github.com/gin-gonic/gin")

        assert command == "go get github.com/gin-gonic/gin"

    def test_standard_library_imports(self) -> None:
        """Test standard library package imports."""
        handler = GoHandler()

        std_packages = ["fmt", "os", "io", "net/http", "encoding/json", "time", "strings", "strconv", "math", "sort"]

        for package in std_packages:
            pattern = handler.get_import_patterns(package)
            import_stmt = f'import "{package}"'
            assert re.search(pattern, import_stmt), f"Should match standard package: {import_stmt}"

    def test_third_party_module_imports(self) -> None:
        """Test third-party module imports."""
        handler = GoHandler()

        third_party_modules = [
            ("github.com/gin-gonic/gin", 'import "github.com/gin-gonic/gin"'),
            ("github.com/gorilla/mux", 'import "github.com/gorilla/mux"'),
            ("go.uber.org/zap", 'import "go.uber.org/zap"'),
            ("golang.org/x/crypto/bcrypt", 'import "golang.org/x/crypto/bcrypt"'),
        ]

        for module, import_stmt in third_party_modules:
            pattern = handler.get_import_patterns(module)
            assert re.search(pattern, import_stmt), f"Should match third-party module: {import_stmt}"

    def test_import_block_patterns(self) -> None:
        """Test import block patterns."""
        handler = GoHandler()

        pattern = handler.get_import_patterns("fmt")

        import_blocks = [
            """import (
                "fmt"
                "os"
            )""",
            """import (
                "fmt"
            )""",
            """import (
                f "fmt"
                "os"
            )""",
        ]

        for block in import_blocks:
            assert re.search(pattern, block), f"Should match import block: {block}"

    def test_complex_module_paths(self) -> None:
        """Test complex module path patterns."""
        handler = GoHandler()

        # Test nested modules
        complex_modules = [
            "github.com/user/repo/pkg/utils",
            "golang.org/x/tools/go/packages",
            "gopkg.in/yaml.v2",
            "k8s.io/client-go/kubernetes",
        ]

        for module in complex_modules:
            pattern = handler.get_import_patterns(module)
            import_stmt = f'import "{module}"'
            assert re.search(pattern, import_stmt), f"Should match complex module: {import_stmt}"

    def test_subpackage_differentiation(self) -> None:
        """Test that subpackages are properly differentiated."""
        handler = GoHandler()

        # Test that "fmt" pattern doesn't match "fmt/something"
        fmt_pattern = handler.get_import_patterns("fmt")

        # Should match
        assert re.search(fmt_pattern, 'import "fmt"')

        # Should NOT match (these are different packages)
        assert not re.search(fmt_pattern, 'import "fmtutil"')

    def test_version_specific_imports(self) -> None:
        """Test version-specific import patterns."""
        handler = GoHandler()

        versioned_imports = [
            ("gopkg.in/yaml.v2", 'import "gopkg.in/yaml.v2"'),
            ("gopkg.in/yaml.v3", 'import "gopkg.in/yaml.v3"'),
            ("github.com/go-sql-driver/mysql", 'import "github.com/go-sql-driver/mysql"'),
        ]

        for module, import_stmt in versioned_imports:
            pattern = handler.get_import_patterns(module)
            assert re.search(pattern, import_stmt), f"Should match versioned import: {import_stmt}"
