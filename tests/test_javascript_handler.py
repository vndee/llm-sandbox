# ruff: noqa: SLF001, PLR2004

import logging
import re
from unittest.mock import MagicMock

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.language_handlers.javascript_handler import JavaScriptHandler


class TestJavaScriptHandler:
    """Test JavaScriptHandler specific functionality."""

    def test_init(self) -> None:
        """Test JavaScriptHandler initialization."""
        handler = JavaScriptHandler()

        assert handler.config.name == SupportedLanguage.JAVASCRIPT
        assert handler.config.file_extension == "js"
        assert "node {file}" in handler.config.execution_commands
        assert handler.config.package_manager == "npm install"
        assert handler.config.plot_detection is None
        assert handler.config.is_support_library_installation is True

    def test_init_with_custom_logger(self) -> None:
        """Test JavaScriptHandler initialization with custom logger."""
        custom_logger = logging.getLogger("custom")
        handler = JavaScriptHandler(custom_logger)
        assert handler.logger == custom_logger

    def test_inject_plot_detection_code(self) -> None:
        """Test plot detection code injection (should return unchanged code)."""
        handler = JavaScriptHandler()
        code = "console.log('Hello World');"

        injected_code = handler.inject_plot_detection_code(code)

        assert injected_code == code  # Should be unchanged since JavaScript doesn't support plot detection

    def test_run_with_artifacts_no_plotting_support(self) -> None:
        """Test run_with_artifacts returns empty plots list."""
        handler = JavaScriptHandler()
        mock_container = MagicMock()
        mock_result = MagicMock()
        mock_container.run.return_value = mock_result

        result, plots = handler.run_with_artifacts(
            container=mock_container,
            code="console.log('Hello');",
            libraries=["axios"],
            enable_plotting=True,
            timeout=30,
            output_dir="/tmp/sandbox_plots",
        )

        assert result == mock_result
        assert plots == []
        mock_container.run.assert_called_once_with("console.log('Hello');", ["axios"], timeout=30)

    def test_extract_plots_returns_empty(self) -> None:
        """Test extract_plots returns empty list."""
        handler = JavaScriptHandler()
        mock_container = MagicMock()

        plots = handler.extract_plots(mock_container, "/tmp/sandbox_plots")

        assert plots == []

    def test_get_import_patterns_es6(self) -> None:
        """Test get_import_patterns method for ES6 imports."""
        handler = JavaScriptHandler()

        pattern = handler.get_import_patterns("axios")

        # Should match ES6 import formats
        es6_imports = [
            "import axios from 'axios';",
            'import axios from "axios";',
            "import { get, post } from 'axios';",
            "import * as axios from 'axios';",
            "import axios, { create } from 'axios';",
        ]

        for code in es6_imports:
            assert re.search(pattern, code), f"Pattern should match ES6 import: {code}"

    def test_get_import_patterns_commonjs(self) -> None:
        """Test get_import_patterns method for CommonJS requires."""
        handler = JavaScriptHandler()

        pattern = handler.get_import_patterns("fs")

        # Should match CommonJS require formats
        commonjs_requires = [
            "const fs = require('fs');",
            'const fs = require("fs");',
            "const { readFile } = require('fs');",
            "var fs = require('fs');",
            "let fs = require('fs');",
            "require('fs')",
        ]

        for code in commonjs_requires:
            assert re.search(pattern, code), f"Pattern should match CommonJS require: {code}"

    def test_get_import_patterns_no_false_positives(self) -> None:
        """Test that import patterns don't match unrelated code."""
        handler = JavaScriptHandler()

        pattern = handler.get_import_patterns("axios")

        # Should not match comments or parts of other words
        non_matching_samples = [
            "// import axios from 'axios';",
            "/* const axios = require('axios'); */",
            "import myaxios from 'myaxios';",  # Different module name
            "const axiosHelper = require('axios-helper');",  # Different module name
            "console.log('import axios from axios');",  # String literal
        ]

        for code in non_matching_samples:
            filtered_code = handler.filter_comments(code)
            if "myaxios" not in code and "axios-helper" not in code and "console.log" not in code:
                assert not re.search(pattern, filtered_code), f"Pattern should not match: {code}"

    def test_get_multiline_comment_patterns(self) -> None:
        """Test get_multiline_comment_patterns method."""
        pattern = JavaScriptHandler.get_multiline_comment_patterns()

        comment_samples = [
            "/* This is a comment */",
            "/*\n * Multiline\n * comment\n */",
            "/* Single line with content */",
            "/**\n * JSDoc comment\n * @param {string} name\n */",
        ]

        for comment in comment_samples:
            assert re.search(pattern, comment), f"Pattern should match: {comment}"

    def test_get_inline_comment_patterns(self) -> None:
        """Test get_inline_comment_patterns method."""
        pattern = JavaScriptHandler.get_inline_comment_patterns()

        comment_samples = [
            "// This is a comment",
            "console.log('Hello');  // Inline comment",
            "    // Indented comment",
            "let x = 5; // Variable definition",
        ]

        for comment in comment_samples:
            assert re.search(pattern, comment), f"Pattern should match: {comment}"

    def test_filter_comments(self) -> None:
        """Test comment filtering functionality."""
        handler = JavaScriptHandler()

        code_with_comments = """
        // This is a single line comment
        function hello() {
            /* This is a
               multiline comment */
            console.log("Hello"); // Inline comment
        }
        """

        filtered_code = handler.filter_comments(code_with_comments)

        # Should remove comments but keep code
        assert 'console.log("Hello");' in filtered_code
        assert "function hello()" in filtered_code
        assert "// This is a single line comment" not in filtered_code
        assert "/* This is a" not in filtered_code
        assert "// Inline comment" not in filtered_code

    def test_properties(self) -> None:
        """Test handler property methods."""
        handler = JavaScriptHandler()

        assert handler.name == SupportedLanguage.JAVASCRIPT
        assert handler.file_extension == "js"
        assert handler.supported_plot_libraries == []
        assert handler.is_support_library_installation is True
        assert handler.is_support_plot_detection is False

    def test_get_execution_commands(self) -> None:
        """Test getting execution commands."""
        handler = JavaScriptHandler()

        commands = handler.get_execution_commands("app.js")

        assert len(commands) == 1
        assert commands[0] == "node app.js"

    def test_get_library_installation_command(self) -> None:
        """Test getting library installation command."""
        handler = JavaScriptHandler()

        command = handler.get_library_installation_command("express")

        assert command == "npm install express"

    def test_complex_import_scenarios(self) -> None:
        """Test complex import scenarios with mixed formats."""
        handler = JavaScriptHandler()

        # Test scoped packages
        pattern = handler.get_import_patterns("@babel/core")

        scoped_imports = [
            "import babel from '@babel/core';",
            "const babel = require('@babel/core');",
            "import { transform } from '@babel/core';",
        ]

        for import_stmt in scoped_imports:
            assert re.search(pattern, import_stmt), f"Should match scoped package: {import_stmt}"

    def test_real_world_imports(self) -> None:
        """Test real-world import patterns."""
        handler = JavaScriptHandler()

        # Test popular JavaScript libraries
        test_cases = [
            ("lodash", "import _ from 'lodash';"),
            ("lodash", "const _ = require('lodash');"),
            ("react", "import React from 'react';"),
            ("react", "const React = require('react');"),
            ("express", "import express from 'express';"),
            ("express", "const express = require('express');"),
        ]

        for module, import_stmt in test_cases:
            pattern = handler.get_import_patterns(module)
            assert re.search(pattern, import_stmt), f"Should match: {import_stmt}"

    def test_destructuring_imports(self) -> None:
        """Test destructuring import patterns."""
        handler = JavaScriptHandler()

        pattern = handler.get_import_patterns("react")

        destructuring_examples = [
            "import { useState, useEffect } from 'react';",
            "import React, { Component } from 'react';",
            "const { useState } = require('react');",
            "const React = require('react');",
        ]

        for example in destructuring_examples:
            assert re.search(pattern, example), f"Should match destructuring: {example}"
