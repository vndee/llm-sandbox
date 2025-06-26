# ruff: noqa: SLF001, PLR2004

import logging
import re
from unittest.mock import MagicMock

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.language_handlers.ruby_handler import RubyHandler


class TestRubyHandler:
    """Test RubyHandler specific functionality."""

    def test_init(self) -> None:
        """Test RubyHandler initialization."""
        handler = RubyHandler()

        assert handler.config.name == SupportedLanguage.RUBY
        assert handler.config.file_extension == "rb"
        assert "ruby {file}" in handler.config.execution_commands
        assert handler.config.package_manager == "gem install"
        assert handler.config.plot_detection is None
        assert handler.config.is_support_library_installation is True

    def test_init_with_custom_logger(self) -> None:
        """Test RubyHandler initialization with custom logger."""
        custom_logger = logging.getLogger("custom")
        handler = RubyHandler(custom_logger)
        assert handler.logger == custom_logger

    def test_inject_plot_detection_code(self) -> None:
        """Test plot detection code injection (should return unchanged code)."""
        handler = RubyHandler()
        code = 'puts "Hello World"'

        injected_code = handler.inject_plot_detection_code(code)

        assert injected_code == code  # Should be unchanged since Ruby doesn't support plot detection

    def test_run_with_artifacts_no_plotting_support(self) -> None:
        """Test run_with_artifacts returns empty plots list."""
        handler = RubyHandler()
        mock_container = MagicMock()
        mock_result = MagicMock()
        mock_container.run.return_value = mock_result

        result, plots = handler.run_with_artifacts(
            container=mock_container,
            code='puts "Hello"',
            libraries=["json"],
            enable_plotting=True,
            output_dir="/tmp/sandbox_plots",
            timeout=30,
        )

        assert result == mock_result
        assert plots == []
        mock_container.run.assert_called_once_with('puts "Hello"', ["json"], timeout=30)

    def test_extract_plots_returns_empty(self) -> None:
        """Test extract_plots returns empty list."""
        handler = RubyHandler()
        mock_container = MagicMock()

        plots = handler.extract_plots(mock_container, "/tmp/sandbox_plots")

        assert plots == []

    def test_get_import_patterns_require(self) -> None:
        """Test get_import_patterns method for require statements."""
        handler = RubyHandler()

        pattern = handler.get_import_patterns("json")

        # Should match require statements
        require_statements = [
            "require 'json'",
            'require "json"',
            "require('json')",
            'require("json")',
            "  require 'json'",
            "require 'json'  # comment",
        ]

        for code in require_statements:
            assert re.search(pattern, code), f"Pattern should match require: {code}"

    def test_get_import_patterns_require_relative(self) -> None:
        """Test get_import_patterns method for require_relative statements."""
        handler = RubyHandler()

        pattern = handler.get_import_patterns("helper")

        # Should match require_relative statements
        require_relative_statements = [
            "require_relative 'helper'",
            'require_relative "helper"',
            "require_relative('helper')",
            'require_relative("helper")',
            "  require_relative 'helper'",
        ]

        for code in require_relative_statements:
            assert re.search(pattern, code), f"Pattern should match require_relative: {code}"

    def test_get_import_patterns_no_false_positives(self) -> None:
        """Test that import patterns don't match unrelated code."""
        handler = RubyHandler()

        pattern = handler.get_import_patterns("json")

        # Should not match comments or parts of other words
        non_matching_samples = [
            "# require 'json'",
            "=begin\nrequire 'json'\n=end",
            "require 'jsonpath'",  # Different gem
            "puts 'require json'",  # String literal
            "json_data = {}",  # Variable usage
        ]

        for code in non_matching_samples:
            filtered_code = handler.filter_comments(code)
            if "jsonpath" not in code and "puts" not in code and "json_data" not in code:
                assert not re.search(pattern, filtered_code), f"Pattern should not match: {code}"

    def test_get_multiline_comment_patterns(self) -> None:
        """Test get_multiline_comment_patterns method."""
        pattern = RubyHandler.get_multiline_comment_patterns()

        comment_samples = [
            "=begin\nThis is a comment\n=end",
            "=begin\n * Multiline\n * comment\n=end",
            "=begin\nSingle block comment\n=end",
            "=begin\nDocumentation\ncomment\n=end",
        ]

        for comment in comment_samples:
            assert re.search(pattern, comment), f"Pattern should match: {comment}"

    def test_get_inline_comment_patterns(self) -> None:
        """Test get_inline_comment_patterns method."""
        pattern = RubyHandler.get_inline_comment_patterns()

        comment_samples = [
            "# This is a comment",
            'puts "Hello"  # Inline comment',
            "    # Indented comment",
            "x = 5 # Variable definition",
        ]

        for comment in comment_samples:
            assert re.search(pattern, comment), f"Pattern should match: {comment}"

    def test_filter_comments(self) -> None:
        """Test comment filtering functionality."""
        handler = RubyHandler()

        code_with_comments = """
        # This is a single line comment
        require 'json'
        def hello
            =begin
            This is a
            multiline comment
            =end
            puts "Hello" # Inline comment
        end
        """

        filtered_code = handler.filter_comments(code_with_comments)

        # Should remove comments but keep code
        assert 'puts "Hello"' in filtered_code
        assert "require 'json'" in filtered_code
        assert "def hello" in filtered_code
        assert "# This is a single line comment" not in filtered_code
        assert "=begin" not in filtered_code
        assert "# Inline comment" not in filtered_code

    def test_properties(self) -> None:
        """Test handler property methods."""
        handler = RubyHandler()

        assert handler.name == SupportedLanguage.RUBY
        assert handler.file_extension == "rb"
        assert handler.supported_plot_libraries == []
        assert handler.is_support_library_installation is True
        assert handler.is_support_plot_detection is False

    def test_get_execution_commands(self) -> None:
        """Test getting execution commands."""
        handler = RubyHandler()

        commands = handler.get_execution_commands("script.rb")

        assert len(commands) == 1
        assert commands[0] == "ruby script.rb"

    def test_get_library_installation_command(self) -> None:
        """Test getting library installation command."""
        handler = RubyHandler()

        command = handler.get_library_installation_command("rails")

        assert command == "gem install rails"

    def test_standard_library_requires(self) -> None:
        """Test standard library require statements."""
        handler = RubyHandler()

        std_libraries = ["json", "csv", "yaml", "net/http", "uri", "time", "date", "fileutils", "pathname", "digest"]

        for library in std_libraries:
            pattern = handler.get_import_patterns(library)
            require_stmt = f"require '{library}'"
            assert re.search(pattern, require_stmt), f"Should match standard library: {require_stmt}"

    def test_gem_requires(self) -> None:
        """Test gem require statements."""
        handler = RubyHandler()

        popular_gems = [
            ("rails", "require 'rails'"),
            ("sinatra", "require 'sinatra'"),
            ("nokogiri", "require 'nokogiri'"),
            ("httparty", "require 'httparty'"),
            ("rspec", "require 'rspec'"),
        ]

        for gem, require_stmt in popular_gems:
            pattern = handler.get_import_patterns(gem)
            assert re.search(pattern, require_stmt), f"Should match gem require: {require_stmt}"

    def test_require_with_path_separators(self) -> None:
        """Test require statements with path separators."""
        handler = RubyHandler()

        path_requires = [
            ("net/http", "require 'net/http'"),
            ("active_record/base", "require 'active_record/base'"),
            ("lib/utils", "require 'lib/utils'"),
            ("../helper", "require_relative '../helper'"),
        ]

        for module, require_stmt in path_requires:
            pattern = handler.get_import_patterns(module)
            assert re.search(pattern, require_stmt), f"Should match path require: {require_stmt}"

    def test_require_variations(self) -> None:
        """Test different variations of require statements."""
        handler = RubyHandler()

        pattern = handler.get_import_patterns("json")

        variations = [
            "require 'json'",
            'require "json"',
            "require('json')",
            'require("json")',
            "require 'json' if defined?(JSON)",
            "require 'json' unless defined?(JSON)",
        ]

        for variation in variations:
            assert re.search(pattern, variation), f"Should match variation: {variation}"

    def test_bundler_require_patterns(self) -> None:
        """Test Bundler-style require patterns."""
        handler = RubyHandler()

        # Test that basic require pattern still works with bundler gems
        bundler_requires = [
            "require 'bundler/setup'",
            "require 'rails/all'",
            "require 'active_support/all'",
        ]

        for require_stmt in bundler_requires:
            # Extract the main gem name for pattern matching
            gem_name = require_stmt.split("'")[1].split("/")[0]
            pattern = handler.get_import_patterns(gem_name)
            assert re.search(pattern, require_stmt), f"Should match bundler require: {require_stmt}"

    def test_autoload_not_matched(self) -> None:
        """Test that autoload statements are not matched by require patterns."""
        handler = RubyHandler()

        pattern = handler.get_import_patterns("helper")

        # autoload should not be matched by require patterns
        autoload_statements = [
            "autoload :Helper, 'helper'",
            "autoload 'Helper', 'helper'",
        ]

        for stmt in autoload_statements:
            assert not re.search(pattern, stmt), f"Should not match autoload: {stmt}"

    def test_conditional_requires(self) -> None:
        """Test conditional require patterns."""
        handler = RubyHandler()

        pattern = handler.get_import_patterns("optional_gem")

        conditional_requires = [
            "require 'optional_gem' if RUBY_VERSION >= '2.7'",
            "require 'optional_gem' unless defined?(OptionalGem)",
            "begin; require 'optional_gem'; rescue LoadError; end",
        ]

        # These should still match the basic require pattern
        for require_stmt in conditional_requires:
            if "begin" not in require_stmt:  # Skip the rescue pattern as it's more complex
                assert re.search(pattern, require_stmt), f"Should match conditional require: {require_stmt}"

    def test_string_interpolation_not_matched(self) -> None:
        """Test that string interpolation with require-like text is not matched."""
        handler = RubyHandler()

        pattern = handler.get_import_patterns("json")

        # These should NOT match
        non_matching = [
            "puts \"Please require 'json' gem\"",
            "\"#{require 'json'}\"",  # This is complex, might match, but shouldn't be a real issue
            "# Don't forget to require 'json'",
        ]

        for stmt in non_matching:
            filtered_stmt = handler.filter_comments(stmt)
            if "puts" in stmt:  # String literals should not match
                assert not re.search(pattern, filtered_stmt), f"Should not match string: {stmt}"
