# documentation.md

This file provides a line-by-line explanation of src/model.py.
Each entry maps one source line to its intent and role.

- Source file: src/model.py
- Total lines explained: 1143

| Line | Code | Explanation |
|---:|---|---|
| 1 | import argparse | Imports module(s) argparse so this file can use their APIs. |
| 2 | import json | Imports module(s) json so this file can use their APIs. |
| 3 | import math | Imports module(s) math so this file can use their APIs. |
| 4 | import random | Imports module(s) random so this file can use their APIs. |
| 5 | import re | Imports module(s) re so this file can use their APIs. |
| 6 | from collections import Counter | Imports symbol(s) Counter from module collections for direct usage. |
| 7 | from dataclasses import dataclass | Imports symbol(s) dataclass from module dataclasses for direct usage. |
| 8 | from datetime import datetime | Imports symbol(s) datetime from module datetime for direct usage. |
| 9 | from pathlib import Path | Imports symbol(s) Path from module pathlib for direct usage. |
| 10 | from typing import Callable, Dict, List, Optional, Sequence, cast | Imports symbol(s) Callable, Dict, List, Optional, Sequence, cast from module typing for direct usage. |
| 11 | &nbsp; | Blank line used to visually separate logical sections. |
| 12 | import numpy as np | Imports module(s) numpy as np so this file can use their APIs. |
| 13 | import pandas as pd | Imports module(s) pandas as pd so this file can use their APIs. |
| 14 | import torch | Imports module(s) torch so this file can use their APIs. |
| 15 | import torch.nn as nn | Imports module(s) torch.nn as nn so this file can use their APIs. |
| 16 | from TorchCRF import CRF | Imports symbol(s) CRF from module TorchCRF for direct usage. |
| 17 | from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support | Imports symbol(s) accuracy_score, cohen_kappa_score, precision_recall_fscore_support from module sklearn.metrics for direct usage. |
| 18 | from sklearn.model_selection import train_test_split | Imports symbol(s) train_test_split from module sklearn.model_selection for direct usage. |
| 19 | from torch.utils.data import DataLoader, Dataset | Imports symbol(s) DataLoader, Dataset from module torch.utils.data for direct usage. |
| 20 | &nbsp; | Blank line used to visually separate logical sections. |
| 21 | &nbsp; | Blank line used to visually separate logical sections. |
| 22 | PRIORITY_LABELS = ["critical", "high", "medium", "low", "unknown"] | Assigns computed value to variable PRIORITY_LABELS for later use. |
| 23 | SEVERITY_LABELS = ["critical", "major", "minor", "trivial", "unknown"] | Assigns computed value to variable SEVERITY_LABELS for later use. |
| 24 | TYPE_LABELS = [ | Assigns computed value to variable TYPE_LABELS for later use. |
| 25 |     "bug", | Continuation line that contributes one element/argument to a larger structure. |
| 26 |     "regression", | Continuation line that contributes one element/argument to a larger structure. |
| 27 |     "feature", | Continuation line that contributes one element/argument to a larger structure. |
| 28 |     "build", | Continuation line that contributes one element/argument to a larger structure. |
| 29 |     "ui", | Continuation line that contributes one element/argument to a larger structure. |
| 30 |     "performance", | Continuation line that contributes one element/argument to a larger structure. |
| 31 |     "security", | Continuation line that contributes one element/argument to a larger structure. |
| 32 |     "unknown", | Continuation line that contributes one element/argument to a larger structure. |
| 33 | ] | Closes a list/dict/function-call block opened on previous lines. |
| 34 | COMPONENT_LABELS = [ | Assigns computed value to variable COMPONENT_LABELS for later use. |
| 35 |     "layout", | Continuation line that contributes one element/argument to a larger structure. |
| 36 |     "javascript", | Continuation line that contributes one element/argument to a larger structure. |
| 37 |     "nspr", | Continuation line that contributes one element/argument to a larger structure. |
| 38 |     "qa", | Continuation line that contributes one element/argument to a larger structure. |
| 39 |     "browser", | Continuation line that contributes one element/argument to a larger structure. |
| 40 |     "networking", | Continuation line that contributes one element/argument to a larger structure. |
| 41 |     "internationalization", | Continuation line that contributes one element/argument to a larger structure. |
| 42 |     "ui", | Continuation line that contributes one element/argument to a larger structure. |
| 43 |     "general", | Continuation line that contributes one element/argument to a larger structure. |
| 44 |     "other", | Continuation line that contributes one element/argument to a larger structure. |
| 45 |     "unknown", | Continuation line that contributes one element/argument to a larger structure. |
| 46 | ] | Closes a list/dict/function-call block opened on previous lines. |
| 47 | &nbsp; | Blank line used to visually separate logical sections. |
| 48 | LABEL_COLUMN_CANDIDATES = ["label", "target", "is_duplicate", "duplicate", "bug_label"] | Assigns computed value to variable LABEL_COLUMN_CANDIDATES for later use. |
| 49 | &nbsp; | Blank line used to visually separate logical sections. |
| 50 | &nbsp; | Blank line used to visually separate logical sections. |
| 51 | def set_seed(seed: int) -&gt; None: | Declares function set_seed and begins its parameter signature. |
| 52 |     """Set all relevant random seeds for reproducible experiments.""" | Single-line docstring that documents behavior for the surrounding function/class/module. |
| 53 |     random.seed(seed) | Continuation line in an ongoing expression, literal, or control-flow block. |
| 54 |     np.random.seed(seed) | Continuation line in an ongoing expression, literal, or control-flow block. |
| 55 |     torch.manual_seed(seed) | Continuation line in an ongoing expression, literal, or control-flow block. |
| 56 |     if torch.cuda.is_available(): | Starts conditional logic; this block executes only when the condition is true. |
| 57 |         torch.cuda.manual_seed_all(seed) | Continuation line in an ongoing expression, literal, or control-flow block. |
| 58 | &nbsp; | Blank line used to visually separate logical sections. |
| 59 | &nbsp; | Blank line used to visually separate logical sections. |
| 60 | def tokenize_text(text: str) -&gt; List[str]: | Declares function tokenize_text and begins its parameter signature. |
| 61 |     """Simple tokenizer used by the ADLM text branch.""" | Single-line docstring that documents behavior for the surrounding function/class/module. |
| 62 |     return re.findall(r"[a-z0-9_]+", str(text).lower()) | Returns a value to the caller from the current function. |
| 63 | &nbsp; | Blank line used to visually separate logical sections. |
| 64 | &nbsp; | Blank line used to visually separate logical sections. |
| 65 | def safe_json_load(value: object) -&gt; Dict[str, object]: | Declares function safe_json_load and begins its parameter signature. |
| 66 |     """Parse JSON-like text safely; returns an empty dict on parse errors.""" | Single-line docstring that documents behavior for the surrounding function/class/module. |
| 67 |     if isinstance(value, dict): | Starts conditional logic; this block executes only when the condition is true. |
| 68 |         return value | Returns a value to the caller from the current function. |
| 69 |     if not isinstance(value, str) or not value.strip(): | Starts conditional logic; this block executes only when the condition is true. |
| 70 |         return {} | Returns a value to the caller from the current function. |
| 71 |     try: | Begins try-block for guarded execution that may raise exceptions. |
| 72 |         parsed = json.loads(value) | Assigns computed value to variable parsed for later use. |
| 73 |         return parsed if isinstance(parsed, dict) else {} | Returns a value to the caller from the current function. |
| 74 |     except json.JSONDecodeError: | Catches and handles exceptions from the preceding try block. |
| 75 |         return {} | Returns a value to the caller from the current function. |
| 76 | &nbsp; | Blank line used to visually separate logical sections. |
| 77 | &nbsp; | Blank line used to visually separate logical sections. |
| 78 | def one_hot(label: str, labels: Sequence[str]) -&gt; List[float]: | Declares function one_hot and begins its parameter signature. |
| 79 |     """Convert a label into a dense one-hot list for fixed-width model input.""" | Single-line docstring that documents behavior for the surrounding function/class/module. |
| 80 |     return [1.0 if label == token else 0.0 for token in labels] | Returns a value to the caller from the current function. |
| 81 | &nbsp; | Blank line used to visually separate logical sections. |
| 82 | &nbsp; | Blank line used to visually separate logical sections. |
| 83 | def infer_priority(text: str) -&gt; str: | Declares function infer_priority and begins its parameter signature. |
| 84 |     lower = text.lower() | Assigns computed value to variable lower for later use. |
| 85 |     explicit = re.search( | Assigns computed value to variable explicit for later use. |
| 86 |         r"\b(?:priority\\|set priority\\|upgrading priority to)\b[^a-z0-9]{0,10}\bp([1-5])\b", | Continuation line that contributes one element/argument to a larger structure. |
| 87 |         lower, | Continuation line that contributes one element/argument to a larger structure. |
| 88 |     ) | Closes a list/dict/function-call block opened on previous lines. |
| 89 |     if explicit: | Starts conditional logic; this block executes only when the condition is true. |
| 90 |         return {"1": "critical", "2": "high", "3": "medium", "4": "low", "5": "low"}[explicit.group(1)] | Returns a value to the caller from the current function. |
| 91 |     if re.search(r"\b(highest\\|critical\\|blocker)\b", lower): | Starts conditional logic; this block executes only when the condition is true. |
| 92 |         return "critical" | Returns a value to the caller from the current function. |
| 93 |     if re.search(r"\bhigh\b", lower): | Starts conditional logic; this block executes only when the condition is true. |
| 94 |         return "high" | Returns a value to the caller from the current function. |
| 95 |     if re.search(r"\b(medium\\|normal)\b", lower): | Starts conditional logic; this block executes only when the condition is true. |
| 96 |         return "medium" | Returns a value to the caller from the current function. |
| 97 |     if re.search(r"\b(low\\|minor\\|trivial)\b", lower): | Starts conditional logic; this block executes only when the condition is true. |
| 98 |         return "low" | Returns a value to the caller from the current function. |
| 99 |     return "unknown" | Returns a value to the caller from the current function. |
| 100 | &nbsp; | Blank line used to visually separate logical sections. |
| 101 | &nbsp; | Blank line used to visually separate logical sections. |
| 102 | def infer_severity(text: str) -&gt; str: | Declares function infer_severity and begins its parameter signature. |
| 103 |     lower = text.lower() | Assigns computed value to variable lower for later use. |
| 104 |     numeric = re.search(r"\bseverity\s*([1-5])\b", lower) | Assigns computed value to variable numeric for later use. |
| 105 |     if numeric: | Starts conditional logic; this block executes only when the condition is true. |
| 106 |         return { | Returns a value to the caller from the current function. |
| 107 |             "1": "critical", | Continuation line that contributes one element/argument to a larger structure. |
| 108 |             "2": "major", | Continuation line that contributes one element/argument to a larger structure. |
| 109 |             "3": "minor", | Continuation line that contributes one element/argument to a larger structure. |
| 110 |             "4": "trivial", | Continuation line that contributes one element/argument to a larger structure. |
| 111 |             "5": "trivial", | Continuation line that contributes one element/argument to a larger structure. |
| 112 |         }[numeric.group(1)] | Continuation line in an ongoing expression, literal, or control-flow block. |
| 113 |     named = re.search(r"\bseverity\b[^a-z0-9]{0,10}(critical\\|major\\|minor\\|trivial)\b", lower) | Assigns computed value to variable named for later use. |
| 114 |     if named: | Starts conditional logic; this block executes only when the condition is true. |
| 115 |         return named.group(1) | Returns a value to the caller from the current function. |
| 116 |     if "critical" in lower: | Starts conditional logic; this block executes only when the condition is true. |
| 117 |         return "critical" | Returns a value to the caller from the current function. |
| 118 |     if "major" in lower: | Starts conditional logic; this block executes only when the condition is true. |
| 119 |         return "major" | Returns a value to the caller from the current function. |
| 120 |     if "minor" in lower: | Starts conditional logic; this block executes only when the condition is true. |
| 121 |         return "minor" | Returns a value to the caller from the current function. |
| 122 |     if "trivial" in lower: | Starts conditional logic; this block executes only when the condition is true. |
| 123 |         return "trivial" | Returns a value to the caller from the current function. |
| 124 |     return "unknown" | Returns a value to the caller from the current function. |
| 125 | &nbsp; | Blank line used to visually separate logical sections. |
| 126 | &nbsp; | Blank line used to visually separate logical sections. |
| 127 | def infer_issue_type(text: str) -&gt; str: | Declares function infer_issue_type and begins its parameter signature. |
| 128 |     lower = text.lower() | Assigns computed value to variable lower for later use. |
| 129 |     if re.search(r"\bregression\b\\|\bused to\b\\|\bno longer\b", lower): | Starts conditional logic; this block executes only when the condition is true. |
| 130 |         return "regression" | Returns a value to the caller from the current function. |
| 131 |     if re.search(r"\bfeature request\b\\|\benhancement\b\\|\badd\b", lower): | Starts conditional logic; this block executes only when the condition is true. |
| 132 |         return "feature" | Returns a value to the caller from the current function. |
| 133 |     if re.search(r"\bcompile\b\\|\bbuild\b\\|\bundefined symbol\b\\|\bmake\b.*\berror\b", lower): | Starts conditional logic; this block executes only when the condition is true. |
| 134 |         return "build" | Returns a value to the caller from the current function. |
| 135 |     if re.search(r"\btoolbar\b\\|\bwindow\b\\|\bmenu\b\\|\bbutton\b\\|\blayout\b", lower): | Starts conditional logic; this block executes only when the condition is true. |
| 136 |         return "ui" | Returns a value to the caller from the current function. |
| 137 |     if re.search(r"\bslow\b\\|\bperformance\b\\|\blatency\b", lower): | Starts conditional logic; this block executes only when the condition is true. |
| 138 |         return "performance" | Returns a value to the caller from the current function. |
| 139 |     if re.search(r"\bsecurity\b\\|\bvulnerability\b\\|\bcve\b", lower): | Starts conditional logic; this block executes only when the condition is true. |
| 140 |         return "security" | Returns a value to the caller from the current function. |
| 141 |     if re.search(r"\bcrash\b\\|\bbug\b\\|\berror\b\\|\bexception\b\\|\bassert\b", lower): | Starts conditional logic; this block executes only when the condition is true. |
| 142 |         return "bug" | Returns a value to the caller from the current function. |
| 143 |     return "unknown" | Returns a value to the caller from the current function. |
| 144 | &nbsp; | Blank line used to visually separate logical sections. |
| 145 | &nbsp; | Blank line used to visually separate logical sections. |
| 146 | def infer_component(text: str) -&gt; str: | Declares function infer_component and begins its parameter signature. |
| 147 |     lower = text.lower() | Assigns computed value to variable lower for later use. |
| 148 |     component_keywords = { | Assigns computed value to variable component_keywords for later use. |
| 149 |         "layout": "layout", | Continuation line that contributes one element/argument to a larger structure. |
| 150 |         "javascript": "javascript", | Continuation line that contributes one element/argument to a larger structure. |
| 151 |         "nspr": "nspr", | Continuation line that contributes one element/argument to a larger structure. |
| 152 |         "qa": "qa", | Continuation line that contributes one element/argument to a larger structure. |
| 153 |         "browser": "browser", | Continuation line that contributes one element/argument to a larger structure. |
| 154 |         "network": "networking", | Continuation line that contributes one element/argument to a larger structure. |
| 155 |         "i18n": "internationalization", | Continuation line that contributes one element/argument to a larger structure. |
| 156 |         "internationalization": "internationalization", | Continuation line that contributes one element/argument to a larger structure. |
| 157 |         "toolbar": "ui", | Continuation line that contributes one element/argument to a larger structure. |
| 158 |         "ui": "ui", | Continuation line that contributes one element/argument to a larger structure. |
| 159 |         "general": "general", | Continuation line that contributes one element/argument to a larger structure. |
| 160 |     } | Closes a list/dict/function-call block opened on previous lines. |
| 161 | &nbsp; | Blank line used to visually separate logical sections. |
| 162 |     for key, value in component_keywords.items(): | Starts an iteration loop over a collection or iterable sequence. |
| 163 |         if f"component to {key}" in lower or f"{key} component" in lower: | Starts conditional logic; this block executes only when the condition is true. |
| 164 |             return value | Returns a value to the caller from the current function. |
| 165 |     for key, value in component_keywords.items(): | Starts an iteration loop over a collection or iterable sequence. |
| 166 |         if re.search(rf"\b{re.escape(key)}\b", lower): | Starts conditional logic; this block executes only when the condition is true. |
| 167 |             return value | Returns a value to the caller from the current function. |
| 168 |     return "unknown" | Returns a value to the caller from the current function. |
| 169 | &nbsp; | Blank line used to visually separate logical sections. |
| 170 | &nbsp; | Blank line used to visually separate logical sections. |
| 171 | def infer_weak_duplicate_label(text: str) -&gt; int: | Declares function infer_weak_duplicate_label and begins its parameter signature. |
| 172 |     """ | Start of a multi-line docstring block. |
| 173 |     Build weak supervision labels if the dataset has no explicit target column. | Continuation line in an ongoing expression, literal, or control-flow block. |
| 174 |     1 =&gt; duplicate report, 0 =&gt; non-duplicate. | Continuation line in an ongoing expression, literal, or control-flow block. |
| 175 |     """ | Start of a multi-line docstring block. |
| 176 |     patterns = [ | Assigns computed value to variable patterns for later use. |
| 177 |         r"\bduplicate of\b", | Continuation line that contributes one element/argument to a larger structure. |
| 178 |         r"\bmarked as a duplicate\b", | Continuation line that contributes one element/argument to a larger structure. |
| 179 |         r"\bmarking dup of\b", | Continuation line that contributes one element/argument to a larger structure. |
| 180 |         r"\bthis bug has been marked as a duplicate\b", | Continuation line that contributes one element/argument to a larger structure. |
| 181 |         r"\bdup of\b", | Continuation line that contributes one element/argument to a larger structure. |
| 182 |         r"\bresolved as duplicate\b", | Continuation line that contributes one element/argument to a larger structure. |
| 183 |     ] | Closes a list/dict/function-call block opened on previous lines. |
| 184 |     lower = text.lower() | Assigns computed value to variable lower for later use. |
| 185 |     return int(any(re.search(pattern, lower) for pattern in patterns)) | Returns a value to the caller from the current function. |
| 186 | &nbsp; | Blank line used to visually separate logical sections. |
| 187 | &nbsp; | Blank line used to visually separate logical sections. |
| 188 | def parse_datetime(raw: str) -&gt; Optional[datetime]: | Declares function parse_datetime and begins its parameter signature. |
| 189 |     """Parse date-time from multiple known bug-report formats.""" | Single-line docstring that documents behavior for the surrounding function/class/module. |
| 190 |     if not raw: | Starts conditional logic; this block executes only when the condition is true. |
| 191 |         return None | Returns a value to the caller from the current function. |
| 192 | &nbsp; | Blank line used to visually separate logical sections. |
| 193 |     # Format from converted corpus_features.csv (e.g., Tuesday April 7 1998 10 57 25 AM PDT) | Comment line documenting intent: Format from converted corpus_features.csv (e.g., Tuesday April 7 1998 10 57 25 AM PDT). |
| 194 |     clean = re.sub(r"\s+", " ", raw).strip() | Assigns computed value to variable clean for later use. |
| 195 |     pattern = re.compile( | Assigns computed value to variable pattern for later use. |
| 196 |         ( | Continuation line in an ongoing expression, literal, or control-flow block. |
| 197 |             r"(?P&lt;weekday&gt;[A-Za-z]+)\s+" | Continuation line in an ongoing expression, literal, or control-flow block. |
| 198 |             r"(?P&lt;month&gt;[A-Za-z]+)\s+" | Continuation line in an ongoing expression, literal, or control-flow block. |
| 199 |             r"(?P&lt;day&gt;\d{1,2})\s+" | Continuation line in an ongoing expression, literal, or control-flow block. |
| 200 |             r"(?P&lt;year&gt;\d{4})\s+" | Continuation line in an ongoing expression, literal, or control-flow block. |
| 201 |             r"(?P&lt;hour&gt;\d{1,2})\s+" | Continuation line in an ongoing expression, literal, or control-flow block. |
| 202 |             r"(?P&lt;minute&gt;\d{1,2})\s+" | Continuation line in an ongoing expression, literal, or control-flow block. |
| 203 |             r"(?P&lt;second&gt;\d{1,2})\s+" | Continuation line in an ongoing expression, literal, or control-flow block. |
| 204 |             r"(?P&lt;ampm&gt;AM\\|PM)" | Continuation line in an ongoing expression, literal, or control-flow block. |
| 205 |         ), | Closes a list/dict/function-call block opened on previous lines. |
| 206 |         re.IGNORECASE, | Continuation line that contributes one element/argument to a larger structure. |
| 207 |     ) | Closes a list/dict/function-call block opened on previous lines. |
| 208 |     match = pattern.search(clean) | Assigns computed value to variable match for later use. |
| 209 |     if match: | Starts conditional logic; this block executes only when the condition is true. |
| 210 |         g = match.groupdict() | Assigns computed value to variable g for later use. |
| 211 |         month_num = datetime.strptime(g["month"][:3], "%b").month | Assigns computed value to variable month_num for later use. |
| 212 |         hour12 = int(g["hour"]) % 12 | Assigns computed value to variable hour12 for later use. |
| 213 |         hour24 = hour12 + (12 if g["ampm"].upper() == "PM" else 0) | Assigns computed value to variable hour24 for later use. |
| 214 |         return datetime( | Returns a value to the caller from the current function. |
| 215 |             year=int(g["year"]), | Assigns computed value to variable year for later use. |
| 216 |             month=month_num, | Assigns computed value to variable month for later use. |
| 217 |             day=int(g["day"]), | Assigns computed value to variable day for later use. |
| 218 |             hour=hour24, | Assigns computed value to variable hour for later use. |
| 219 |             minute=int(g["minute"]), | Assigns computed value to variable minute for later use. |
| 220 |             second=int(g["second"]), | Assigns computed value to variable second for later use. |
| 221 |         ) | Closes a list/dict/function-call block opened on previous lines. |
| 222 | &nbsp; | Blank line used to visually separate logical sections. |
| 223 |     # Format from Jira-like exports (e.g., 29/May/2023 6:43 AM) | Comment line documenting intent: Format from Jira-like exports (e.g., 29/May/2023 6:43 AM). |
| 224 |     for fmt in ["%d/%b/%Y %I:%M %p", "%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M:%S"]: | Starts an iteration loop over a collection or iterable sequence. |
| 225 |         try: | Begins try-block for guarded execution that may raise exceptions. |
| 226 |             return datetime.strptime(clean, fmt) | Returns a value to the caller from the current function. |
| 227 |         except ValueError: | Catches and handles exceptions from the preceding try block. |
| 228 |             continue | Continuation line in an ongoing expression, literal, or control-flow block. |
| 229 |     return None | Returns a value to the caller from the current function. |
| 230 | &nbsp; | Blank line used to visually separate logical sections. |
| 231 | &nbsp; | Blank line used to visually separate logical sections. |
| 232 | def build_temporal_vector(row: pd.Series) -&gt; List[float]: | Declares function build_temporal_vector and begins its parameter signature. |
| 233 |     """ | Start of a multi-line docstring block. |
| 234 |     Convert time values into a numeric vector. | Continuation line in an ongoing expression, literal, or control-flow block. |
| 235 |     Uses cyclic encoding for calendar/time features to preserve periodicity. | Continuation line in an ongoing expression, literal, or control-flow block. |
| 236 |     """ | Start of a multi-line docstring block. |
| 237 |     payload = safe_json_load(row.get("temporal_features", "")) | Assigns computed value to variable payload for later use. |
| 238 |     created_raw = str(payload.get("created_raw", "")) if payload else "" | Assigns computed value to variable created_raw for later use. |
| 239 |     if not created_raw and "Created" in row: | Starts conditional logic; this block executes only when the condition is true. |
| 240 |         created_raw = str(row.get("Created", "")) | Assigns computed value to variable created_raw for later use. |
| 241 | &nbsp; | Blank line used to visually separate logical sections. |
| 242 |     dt = parse_datetime(created_raw) | Assigns computed value to variable dt for later use. |
| 243 |     if dt is None: | Starts conditional logic; this block executes only when the condition is true. |
| 244 |         return [0.0] * 9 | Returns a value to the caller from the current function. |
| 245 | &nbsp; | Blank line used to visually separate logical sections. |
| 246 |     year_scaled = (dt.year - 1990) / 50.0 | Assigns computed value to variable year_scaled for later use. |
| 247 |     month_angle = 2.0 * math.pi * dt.month / 12.0 | Assigns computed value to variable month_angle for later use. |
| 248 |     weekday_angle = 2.0 * math.pi * dt.weekday() / 7.0 | Assigns computed value to variable weekday_angle for later use. |
| 249 |     hour_angle = 2.0 * math.pi * dt.hour / 24.0 | Assigns computed value to variable hour_angle for later use. |
| 250 |     return [ | Returns a value to the caller from the current function. |
| 251 |         year_scaled, | Continuation line that contributes one element/argument to a larger structure. |
| 252 |         math.sin(month_angle), | Continuation line that contributes one element/argument to a larger structure. |
| 253 |         math.cos(month_angle), | Continuation line that contributes one element/argument to a larger structure. |
| 254 |         math.sin(weekday_angle), | Continuation line that contributes one element/argument to a larger structure. |
| 255 |         math.cos(weekday_angle), | Continuation line that contributes one element/argument to a larger structure. |
| 256 |         math.sin(hour_angle), | Continuation line that contributes one element/argument to a larger structure. |
| 257 |         math.cos(hour_angle), | Continuation line that contributes one element/argument to a larger structure. |
| 258 |         dt.minute / 59.0, | Continuation line that contributes one element/argument to a larger structure. |
| 259 |         float(dt.weekday() &gt;= 5), | Function or method call that triggers computation or side effects. |
| 260 |     ] | Closes a list/dict/function-call block opened on previous lines. |
| 261 | &nbsp; | Blank line used to visually separate logical sections. |
| 262 | &nbsp; | Blank line used to visually separate logical sections. |
| 263 | def build_categorical_vector(row: pd.Series, fallback_text: str) -&gt; List[float]: | Declares function build_categorical_vector and begins its parameter signature. |
| 264 |     """Build a fixed-width categorical vector from JSON features or fallback heuristics.""" | Single-line docstring that documents behavior for the surrounding function/class/module. |
| 265 |     payload = safe_json_load(row.get("categorical_features", "")) | Assigns computed value to variable payload for later use. |
| 266 |     vector_obj = payload.get("vector") if payload else None | Assigns computed value to variable vector_obj for later use. |
| 267 |     if isinstance(vector_obj, list): | Starts conditional logic; this block executes only when the condition is true. |
| 268 |         return [float(value) for value in vector_obj] | Returns a value to the caller from the current function. |
| 269 | &nbsp; | Blank line used to visually separate logical sections. |
| 270 |     priority = str(payload.get("priority", "")) if payload else "" | Assigns computed value to variable priority for later use. |
| 271 |     severity = str(payload.get("severity", "")) if payload else "" | Assigns computed value to variable severity for later use. |
| 272 |     issue_type = str(payload.get("issue_type", "")) if payload else "" | Assigns computed value to variable issue_type for later use. |
| 273 |     component = str(payload.get("component", "")) if payload else "" | Assigns computed value to variable component for later use. |
| 274 | &nbsp; | Blank line used to visually separate logical sections. |
| 275 |     # Fall back to paper-inspired categorical extraction when structured fields are missing. | Comment line documenting intent: Fall back to paper-inspired categorical extraction when structured fields are missing.. |
| 276 |     merged = " ".join( | Assigns computed value to variable merged for later use. |
| 277 |         [ | Continuation line in an ongoing expression, literal, or control-flow block. |
| 278 |             fallback_text, | Continuation line that contributes one element/argument to a larger structure. |
| 279 |             str(row.get("Priority", "")), | Function or method call that triggers computation or side effects. |
| 280 |             str(row.get("Severity", "")), | Function or method call that triggers computation or side effects. |
| 281 |             str(row.get("Issue Type", "")), | Function or method call that triggers computation or side effects. |
| 282 |             str(row.get("Component/s", "")), | Function or method call that triggers computation or side effects. |
| 283 |         ] | Closes a list/dict/function-call block opened on previous lines. |
| 284 |     ) | Closes a list/dict/function-call block opened on previous lines. |
| 285 |     if not priority: | Starts conditional logic; this block executes only when the condition is true. |
| 286 |         priority = infer_priority(merged) | Assigns computed value to variable priority for later use. |
| 287 |     if not severity: | Starts conditional logic; this block executes only when the condition is true. |
| 288 |         severity = infer_severity(merged) | Assigns computed value to variable severity for later use. |
| 289 |     if not issue_type: | Starts conditional logic; this block executes only when the condition is true. |
| 290 |         issue_type = infer_issue_type(merged) | Assigns computed value to variable issue_type for later use. |
| 291 |     if not component: | Starts conditional logic; this block executes only when the condition is true. |
| 292 |         component = infer_component(merged) | Assigns computed value to variable component for later use. |
| 293 | &nbsp; | Blank line used to visually separate logical sections. |
| 294 |     vector = ( | Assigns computed value to variable vector for later use. |
| 295 |         one_hot(priority, PRIORITY_LABELS) | Function or method call that triggers computation or side effects. |
| 296 |         + one_hot(severity, SEVERITY_LABELS) | Continuation line in an ongoing expression, literal, or control-flow block. |
| 297 |         + one_hot(issue_type, TYPE_LABELS) | Continuation line in an ongoing expression, literal, or control-flow block. |
| 298 |         + one_hot(component, COMPONENT_LABELS) | Continuation line in an ongoing expression, literal, or control-flow block. |
| 299 |     ) | Closes a list/dict/function-call block opened on previous lines. |
| 300 |     return [float(v) for v in vector] | Returns a value to the caller from the current function. |
| 301 | &nbsp; | Blank line used to visually separate logical sections. |
| 302 | &nbsp; | Blank line used to visually separate logical sections. |
| 303 | def build_contextual_vector(row: pd.Series) -&gt; List[float]: | Declares function build_contextual_vector and begins its parameter signature. |
| 304 |     """ | Start of a multi-line docstring block. |
| 305 |     Extract context cues from logs/code snippets/error traces. | Continuation line in an ongoing expression, literal, or control-flow block. |
| 306 |     This keeps the contextual branch compact and robust. | Continuation line in an ongoing expression, literal, or control-flow block. |
| 307 |     """ | Start of a multi-line docstring block. |
| 308 |     contextual_text = str(row.get("contextual_features", "")).strip() | Assigns computed value to variable contextual_text for later use. |
| 309 |     if not contextual_text: | Starts conditional logic; this block executes only when the condition is true. |
| 310 |         contextual_text = str(row.get("Description", "")).strip() | Assigns computed value to variable contextual_text for later use. |
| 311 |     lower = contextual_text.lower() | Assigns computed value to variable lower for later use. |
| 312 | &nbsp; | Blank line used to visually separate logical sections. |
| 313 |     has_stack_trace = bool(re.search(r"\b(stack trace\\|traceback\\|call stack)\b", lower)) | Assigns computed value to variable has_stack_trace for later use. |
| 314 |     has_error = bool(re.search(r"\b(error\\|exception\\|fatal\\|assert\\|segmentation fault\\|sigsegv)\b", lower)) | Assigns computed value to variable has_error for later use. |
| 315 |     has_code = bool( | Assigns computed value to variable has_code for later use. |
| 316 |         re.search(r"\b0x[0-9a-f]{5,}\b", lower) | Continuation line in an ongoing expression, literal, or control-flow block. |
| 317 |         or re.search(r"\b[A-Za-z_][\w\-.]*\.(c\\|cc\\|cpp\\|h\\|hpp\\|java\\|js\\|py\\|cs\\|go)\b", contextual_text) | Continuation line in an ongoing expression, literal, or control-flow block. |
| 318 |         or re.search(r"[{}();]", contextual_text) | Continuation line in an ongoing expression, literal, or control-flow block. |
| 319 |     ) | Closes a list/dict/function-call block opened on previous lines. |
| 320 |     context_token_count = len(tokenize_text(contextual_text)) | Assigns computed value to variable context_token_count for later use. |
| 321 |     normalized_length = min(context_token_count / 300.0, 1.0) | Assigns computed value to variable normalized_length for later use. |
| 322 | &nbsp; | Blank line used to visually separate logical sections. |
| 323 |     return [ | Returns a value to the caller from the current function. |
| 324 |         float(has_stack_trace), | Function or method call that triggers computation or side effects. |
| 325 |         float(has_error), | Function or method call that triggers computation or side effects. |
| 326 |         float(has_code), | Function or method call that triggers computation or side effects. |
| 327 |         normalized_length, | Continuation line that contributes one element/argument to a larger structure. |
| 328 |     ] | Closes a list/dict/function-call block opened on previous lines. |
| 329 | &nbsp; | Blank line used to visually separate logical sections. |
| 330 | &nbsp; | Blank line used to visually separate logical sections. |
| 331 | @dataclass | Applies @dataclass to auto-generate constructor and utility methods for the next class. |
| 332 | class BugExample: | Declares class BugExample, creating a reusable type in the modeling pipeline. |
| 333 |     """Container for a single bug report sample with all feature branches.""" | Single-line docstring that documents behavior for the surrounding function/class/module. |
| 334 | &nbsp; | Blank line used to visually separate logical sections. |
| 335 |     text: str | Continuation line in an ongoing expression, literal, or control-flow block. |
| 336 |     temporal_vector: List[float] | Continuation line in an ongoing expression, literal, or control-flow block. |
| 337 |     categorical_vector: List[float] | Continuation line in an ongoing expression, literal, or control-flow block. |
| 338 |     contextual_vector: List[float] | Continuation line in an ongoing expression, literal, or control-flow block. |
| 339 |     label: int | Continuation line in an ongoing expression, literal, or control-flow block. |
| 340 | &nbsp; | Blank line used to visually separate logical sections. |
| 341 | &nbsp; | Blank line used to visually separate logical sections. |
| 342 | class Vocabulary: | Declares class Vocabulary, creating a reusable type in the modeling pipeline. |
| 343 |     """Vocabulary for text-to-id conversion in the embedding branch.""" | Single-line docstring that documents behavior for the surrounding function/class/module. |
| 344 | &nbsp; | Blank line used to visually separate logical sections. |
| 345 |     PAD = "&lt;pad&gt;" | Assigns computed value to variable PAD for later use. |
| 346 |     UNK = "&lt;unk&gt;" | Assigns computed value to variable UNK for later use. |
| 347 | &nbsp; | Blank line used to visually separate logical sections. |
| 348 |     def __init__(self) -&gt; None: | Declares function __init__ and begins its parameter signature. |
| 349 |         self.token_to_id: Dict[str, int] = {self.PAD: 0, self.UNK: 1} | Continuation line in an ongoing expression, literal, or control-flow block. |
| 350 |         self.id_to_token: List[str] = [self.PAD, self.UNK] | Continuation line in an ongoing expression, literal, or control-flow block. |
| 351 | &nbsp; | Blank line used to visually separate logical sections. |
| 352 |     def build(self, texts: Sequence[str], max_size: int) -&gt; None: | Declares function build and begins its parameter signature. |
| 353 |         counter: Counter[str] = Counter() | Continuation line in an ongoing expression, literal, or control-flow block. |
| 354 |         for text in texts: | Starts an iteration loop over a collection or iterable sequence. |
| 355 |             counter.update(tokenize_text(text)) | Continuation line in an ongoing expression, literal, or control-flow block. |
| 356 | &nbsp; | Blank line used to visually separate logical sections. |
| 357 |         for token, _ in counter.most_common(max_size - len(self.token_to_id)): | Starts an iteration loop over a collection or iterable sequence. |
| 358 |             if token not in self.token_to_id: | Starts conditional logic; this block executes only when the condition is true. |
| 359 |                 self.token_to_id[token] = len(self.id_to_token) | Continuation line in an ongoing expression, literal, or control-flow block. |
| 360 |                 self.id_to_token.append(token) | Continuation line in an ongoing expression, literal, or control-flow block. |
| 361 | &nbsp; | Blank line used to visually separate logical sections. |
| 362 |     def encode(self, text: str, max_seq_len: int) -&gt; Dict[str, torch.Tensor]: | Declares function encode and begins its parameter signature. |
| 363 |         tokens = tokenize_text(text) | Assigns computed value to variable tokens for later use. |
| 364 |         token_ids = [self.token_to_id.get(token, self.token_to_id[self.UNK]) for token in tokens] | Assigns computed value to variable token_ids for later use. |
| 365 | &nbsp; | Blank line used to visually separate logical sections. |
| 366 |         # Truncate or pad to fixed sequence length for efficient batching. | Comment line documenting intent: Truncate or pad to fixed sequence length for efficient batching.. |
| 367 |         token_ids = token_ids[:max_seq_len] | Assigns computed value to variable token_ids for later use. |
| 368 |         attention_mask = [1] * len(token_ids) | Assigns computed value to variable attention_mask for later use. |
| 369 |         if len(token_ids) &lt; max_seq_len: | Starts conditional logic; this block executes only when the condition is true. |
| 370 |             pad_count = max_seq_len - len(token_ids) | Assigns computed value to variable pad_count for later use. |
| 371 |             token_ids.extend([self.token_to_id[self.PAD]] * pad_count) | Continuation line in an ongoing expression, literal, or control-flow block. |
| 372 |             attention_mask.extend([0] * pad_count) | Continuation line in an ongoing expression, literal, or control-flow block. |
| 373 | &nbsp; | Blank line used to visually separate logical sections. |
| 374 |         return { | Returns a value to the caller from the current function. |
| 375 |             "input_ids": torch.tensor(token_ids, dtype=torch.long), | Continuation line that contributes one element/argument to a larger structure. |
| 376 |             "attention_mask": torch.tensor(attention_mask, dtype=torch.bool), | Continuation line that contributes one element/argument to a larger structure. |
| 377 |         } | Closes a list/dict/function-call block opened on previous lines. |
| 378 | &nbsp; | Blank line used to visually separate logical sections. |
| 379 |     @property | Continuation line in an ongoing expression, literal, or control-flow block. |
| 380 |     def size(self) -&gt; int: | Declares function size and begins its parameter signature. |
| 381 |         return len(self.id_to_token) | Returns a value to the caller from the current function. |
| 382 | &nbsp; | Blank line used to visually separate logical sections. |
| 383 | &nbsp; | Blank line used to visually separate logical sections. |
| 384 | def load_modeling_dataframe(data_path: Path) -&gt; pd.DataFrame: | Declares function load_modeling_dataframe and begins its parameter signature. |
| 385 |     """Load CSV and adapt columns so ADLM can be trained from multiple formats.""" | Single-line docstring that documents behavior for the surrounding function/class/module. |
| 386 |     df = pd.read_csv(data_path) | Assigns computed value to variable df for later use. |
| 387 |     df = df.copy() | Assigns computed value to variable df for later use. |
| 388 | &nbsp; | Blank line used to visually separate logical sections. |
| 389 |     # If the file is the converted corpus output, these columns already exist. | Comment line documenting intent: If the file is the converted corpus output, these columns already exist.. |
| 390 |     if "textual_features" not in df.columns: | Starts conditional logic; this block executes only when the condition is true. |
| 391 |         summary = df.get("Summary", pd.Series([""] * len(df))).fillna("") | Assigns computed value to variable summary for later use. |
| 392 |         description = df.get("Description", pd.Series([""] * len(df))).fillna("") | Assigns computed value to variable description for later use. |
| 393 |         df["textual_features"] = (summary.astype(str) + " " + description.astype(str)).str.strip() | Continuation line in an ongoing expression, literal, or control-flow block. |
| 394 | &nbsp; | Blank line used to visually separate logical sections. |
| 395 |     if "contextual_features" not in df.columns: | Starts conditional logic; this block executes only when the condition is true. |
| 396 |         df["contextual_features"] = df.get("Description", pd.Series([""] * len(df))).fillna("") | Continuation line in an ongoing expression, literal, or control-flow block. |
| 397 | &nbsp; | Blank line used to visually separate logical sections. |
| 398 |     return df | Returns a value to the caller from the current function. |
| 399 | &nbsp; | Blank line used to visually separate logical sections. |
| 400 | &nbsp; | Blank line used to visually separate logical sections. |
| 401 | def detect_label_column(df: pd.DataFrame, explicit_label_column: Optional[str]) -&gt; Optional[str]: | Declares function detect_label_column and begins its parameter signature. |
| 402 |     """Resolve the supervision column if it exists in the dataset.""" | Single-line docstring that documents behavior for the surrounding function/class/module. |
| 403 |     if explicit_label_column and explicit_label_column in df.columns: | Starts conditional logic; this block executes only when the condition is true. |
| 404 |         return explicit_label_column | Returns a value to the caller from the current function. |
| 405 |     for candidate in LABEL_COLUMN_CANDIDATES: | Starts an iteration loop over a collection or iterable sequence. |
| 406 |         if candidate in df.columns: | Starts conditional logic; this block executes only when the condition is true. |
| 407 |             return candidate | Returns a value to the caller from the current function. |
| 408 |     return None | Returns a value to the caller from the current function. |
| 409 | &nbsp; | Blank line used to visually separate logical sections. |
| 410 | &nbsp; | Blank line used to visually separate logical sections. |
| 411 | def normalize_label(value: object) -&gt; Optional[int]: | Declares function normalize_label and begins its parameter signature. |
| 412 |     """Normalize labels from numeric/string columns into 0/1.""" | Single-line docstring that documents behavior for the surrounding function/class/module. |
| 413 |     if value is None: | Starts conditional logic; this block executes only when the condition is true. |
| 414 |         return None | Returns a value to the caller from the current function. |
| 415 |     if isinstance(value, (float, np.floating)) and math.isnan(float(value)): | Starts conditional logic; this block executes only when the condition is true. |
| 416 |         return None | Returns a value to the caller from the current function. |
| 417 |     if isinstance(value, (int, float)): | Starts conditional logic; this block executes only when the condition is true. |
| 418 |         return int(float(value) &gt; 0) | Returns a value to the caller from the current function. |
| 419 | &nbsp; | Blank line used to visually separate logical sections. |
| 420 |     text = str(value).strip().lower() | Assigns computed value to variable text for later use. |
| 421 |     truthy = {"1", "true", "yes", "duplicate", "dup", "y"} | Assigns computed value to variable truthy for later use. |
| 422 |     falsy = {"0", "false", "no", "non-duplicate", "not duplicate", "n"} | Assigns computed value to variable falsy for later use. |
| 423 |     if text in truthy: | Starts conditional logic; this block executes only when the condition is true. |
| 424 |         return 1 | Returns a value to the caller from the current function. |
| 425 |     if text in falsy: | Starts conditional logic; this block executes only when the condition is true. |
| 426 |         return 0 | Returns a value to the caller from the current function. |
| 427 | &nbsp; | Blank line used to visually separate logical sections. |
| 428 |     # Last fallback for labels like "Severity 3 - Minor" etc. | Comment line documenting intent: Last fallback for labels like "Severity 3 - Minor" etc.. |
| 429 |     if text.isdigit(): | Starts conditional logic; this block executes only when the condition is true. |
| 430 |         return int(int(text) &gt; 0) | Returns a value to the caller from the current function. |
| 431 |     return None | Returns a value to the caller from the current function. |
| 432 | &nbsp; | Blank line used to visually separate logical sections. |
| 433 | &nbsp; | Blank line used to visually separate logical sections. |
| 434 | def build_examples(df: pd.DataFrame, label_column: Optional[str]) -&gt; List[BugExample]: | Declares function build_examples and begins its parameter signature. |
| 435 |     """ | Start of a multi-line docstring block. |
| 436 |     Convert dataframe rows into model-ready examples. | Continuation line in an ongoing expression, literal, or control-flow block. |
| 437 |     If no target column exists, weak labels are inferred from duplicate keywords. | Starts conditional logic; this block executes only when the condition is true. |
| 438 |     """ | Start of a multi-line docstring block. |
| 439 |     examples: List[BugExample] = [] | Continuation line in an ongoing expression, literal, or control-flow block. |
| 440 | &nbsp; | Blank line used to visually separate logical sections. |
| 441 |     for _, row in df.iterrows(): | Starts an iteration loop over a collection or iterable sequence. |
| 442 |         text = str(row.get("textual_features", "")).strip() | Assigns computed value to variable text for later use. |
| 443 |         if not text: | Starts conditional logic; this block executes only when the condition is true. |
| 444 |             continue | Continuation line in an ongoing expression, literal, or control-flow block. |
| 445 | &nbsp; | Blank line used to visually separate logical sections. |
| 446 |         contextual_text = str(row.get("contextual_features", "")) | Assigns computed value to variable contextual_text for later use. |
| 447 |         label: Optional[int] = None | Continuation line in an ongoing expression, literal, or control-flow block. |
| 448 |         if label_column: | Starts conditional logic; this block executes only when the condition is true. |
| 449 |             label = normalize_label(row.get(label_column)) | Assigns computed value to variable label for later use. |
| 450 | &nbsp; | Blank line used to visually separate logical sections. |
| 451 |         if label is None: | Starts conditional logic; this block executes only when the condition is true. |
| 452 |             # Weak supervision fallback keeps the training script runnable on unlabeled corpora. | Comment line documenting intent: Weak supervision fallback keeps the training script runnable on unlabeled corpora.. |
| 453 |             label = infer_weak_duplicate_label(text + " " + contextual_text) | Assigns computed value to variable label for later use. |
| 454 | &nbsp; | Blank line used to visually separate logical sections. |
| 455 |         examples.append( | Continuation line in an ongoing expression, literal, or control-flow block. |
| 456 |             BugExample( | Function or method call that triggers computation or side effects. |
| 457 |                 text=text, | Assigns computed value to variable text for later use. |
| 458 |                 temporal_vector=build_temporal_vector(row), | Assigns computed value to variable temporal_vector for later use. |
| 459 |                 categorical_vector=build_categorical_vector(row, text + " " + contextual_text), | Assigns computed value to variable categorical_vector for later use. |
| 460 |                 contextual_vector=build_contextual_vector(row), | Assigns computed value to variable contextual_vector for later use. |
| 461 |                 label=int(label), | Assigns computed value to variable label for later use. |
| 462 |             ) | Closes a list/dict/function-call block opened on previous lines. |
| 463 |         ) | Closes a list/dict/function-call block opened on previous lines. |
| 464 | &nbsp; | Blank line used to visually separate logical sections. |
| 465 |     return examples | Returns a value to the caller from the current function. |
| 466 | &nbsp; | Blank line used to visually separate logical sections. |
| 467 | &nbsp; | Blank line used to visually separate logical sections. |
| 468 | class BugReportDataset(Dataset): | Declares class BugReportDataset, creating a reusable type in the modeling pipeline. |
| 469 |     """PyTorch dataset that yields all ADLM branches per sample.""" | Single-line docstring that documents behavior for the surrounding function/class/module. |
| 470 | &nbsp; | Blank line used to visually separate logical sections. |
| 471 |     def __init__(self, examples: Sequence[BugExample], vocabulary: Vocabulary, max_seq_len: int) -&gt; None: | Declares function __init__ and begins its parameter signature. |
| 472 |         self.examples = list(examples) | Assigns computed value to variable self.examples for later use. |
| 473 |         self.vocabulary = vocabulary | Assigns computed value to variable self.vocabulary for later use. |
| 474 |         self.max_seq_len = max_seq_len | Assigns computed value to variable self.max_seq_len for later use. |
| 475 | &nbsp; | Blank line used to visually separate logical sections. |
| 476 |         # Ensure fixed vector width across the full split. | Comment line documenting intent: Ensure fixed vector width across the full split.. |
| 477 |         self.temporal_dim = len(self.examples[0].temporal_vector) if self.examples else 0 | Assigns computed value to variable self.temporal_dim for later use. |
| 478 |         self.categorical_dim = len(self.examples[0].categorical_vector) if self.examples else 0 | Assigns computed value to variable self.categorical_dim for later use. |
| 479 |         self.contextual_dim = len(self.examples[0].contextual_vector) if self.examples else 0 | Assigns computed value to variable self.contextual_dim for later use. |
| 480 | &nbsp; | Blank line used to visually separate logical sections. |
| 481 |     def __len__(self) -&gt; int: | Declares function __len__ and begins its parameter signature. |
| 482 |         return len(self.examples) | Returns a value to the caller from the current function. |
| 483 | &nbsp; | Blank line used to visually separate logical sections. |
| 484 |     def __getitem__(self, index: int) -&gt; Dict[str, torch.Tensor]: | Declares function __getitem__ and begins its parameter signature. |
| 485 |         example = self.examples[index] | Assigns computed value to variable example for later use. |
| 486 |         encoded = self.vocabulary.encode(example.text, self.max_seq_len) | Assigns computed value to variable encoded for later use. |
| 487 | &nbsp; | Blank line used to visually separate logical sections. |
| 488 |         label = torch.tensor(example.label, dtype=torch.long) | Assigns computed value to variable label for later use. |
| 489 | &nbsp; | Blank line used to visually separate logical sections. |
| 490 |         # CRF expects token-level labels; we broadcast the report label over non-padding tokens. | Comment line documenting intent: CRF expects token-level labels; we broadcast the report label over non-padding tokens.. |
| 491 |         tag_ids = torch.full((self.max_seq_len,), fill_value=example.label, dtype=torch.long) | Assigns computed value to variable tag_ids for later use. |
| 492 |         tag_ids = tag_ids * encoded["attention_mask"].long() | Assigns computed value to variable tag_ids for later use. |
| 493 | &nbsp; | Blank line used to visually separate logical sections. |
| 494 |         return { | Returns a value to the caller from the current function. |
| 495 |             "input_ids": encoded["input_ids"], | Continuation line that contributes one element/argument to a larger structure. |
| 496 |             "attention_mask": encoded["attention_mask"], | Continuation line that contributes one element/argument to a larger structure. |
| 497 |             "temporal_features": torch.tensor(example.temporal_vector, dtype=torch.float32), | Continuation line that contributes one element/argument to a larger structure. |
| 498 |             "categorical_features": torch.tensor(example.categorical_vector, dtype=torch.float32), | Continuation line that contributes one element/argument to a larger structure. |
| 499 |             "contextual_features": torch.tensor(example.contextual_vector, dtype=torch.float32), | Continuation line that contributes one element/argument to a larger structure. |
| 500 |             "label": label, | Continuation line that contributes one element/argument to a larger structure. |
| 501 |             "tag_ids": tag_ids, | Continuation line that contributes one element/argument to a larger structure. |
| 502 |         } | Closes a list/dict/function-call block opened on previous lines. |
| 503 | &nbsp; | Blank line used to visually separate logical sections. |
| 504 | &nbsp; | Blank line used to visually separate logical sections. |
| 505 | class ADLMBiLSTMCRF(nn.Module): | Declares class ADLMBiLSTMCRF, creating a reusable type in the modeling pipeline. |
| 506 |     """ | Start of a multi-line docstring block. |
| 507 |     ADLM-inspired architecture from the paper: | Continuation line in an ongoing expression, literal, or control-flow block. |
| 508 |     text embedding + BiLSTM + fused temporal/categorical/contextual features + CRF. | Continuation line in an ongoing expression, literal, or control-flow block. |
| 509 |     """ | Start of a multi-line docstring block. |
| 510 | &nbsp; | Blank line used to visually separate logical sections. |
| 511 |     def __init__( | Declares function __init__ and begins its parameter signature. |
| 512 |         self, | Continuation line that contributes one element/argument to a larger structure. |
| 513 |         vocab_size: int, | Continuation line that contributes one element/argument to a larger structure. |
| 514 |         embedding_dim: int, | Continuation line that contributes one element/argument to a larger structure. |
| 515 |         hidden_size: int, | Continuation line that contributes one element/argument to a larger structure. |
| 516 |         dropout: float, | Continuation line that contributes one element/argument to a larger structure. |
| 517 |         temporal_dim: int, | Continuation line that contributes one element/argument to a larger structure. |
| 518 |         categorical_dim: int, | Continuation line that contributes one element/argument to a larger structure. |
| 519 |         contextual_dim: int, | Continuation line that contributes one element/argument to a larger structure. |
| 520 |         num_labels: int = 2, | Continuation line that contributes one element/argument to a larger structure. |
| 521 |         crf_regularization: float = 1e-4, | Continuation line that contributes one element/argument to a larger structure. |
| 522 |     ) -&gt; None: | Continuation line in an ongoing expression, literal, or control-flow block. |
| 523 |         super().__init__() | Function or method call that triggers computation or side effects. |
| 524 |         self.hidden_size = hidden_size | Assigns computed value to variable self.hidden_size for later use. |
| 525 |         self.num_labels = num_labels | Assigns computed value to variable self.num_labels for later use. |
| 526 |         self.crf_regularization = crf_regularization | Assigns computed value to variable self.crf_regularization for later use. |
| 527 | &nbsp; | Blank line used to visually separate logical sections. |
| 528 |         self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0) | Assigns computed value to variable self.embedding for later use. |
| 529 |         self.text_encoder = nn.LSTM( | Assigns computed value to variable self.text_encoder for later use. |
| 530 |             input_size=embedding_dim, | Assigns computed value to variable input_size for later use. |
| 531 |             hidden_size=hidden_size, | Assigns computed value to variable hidden_size for later use. |
| 532 |             batch_first=True, | Assigns computed value to variable batch_first for later use. |
| 533 |             bidirectional=True, | Assigns computed value to variable bidirectional for later use. |
| 534 |         ) | Closes a list/dict/function-call block opened on previous lines. |
| 535 |         self.dropout = nn.Dropout(dropout) | Assigns computed value to variable self.dropout for later use. |
| 536 | &nbsp; | Blank line used to visually separate logical sections. |
| 537 |         side_input_dim = temporal_dim + categorical_dim + contextual_dim | Assigns computed value to variable side_input_dim for later use. |
| 538 |         # Side-channel MLP aligns feature scales before fusion with token states. | Comment line documenting intent: Side-channel MLP aligns feature scales before fusion with token states.. |
| 539 |         self.side_encoder = nn.Sequential( | Assigns computed value to variable self.side_encoder for later use. |
| 540 |             nn.Linear(max(1, side_input_dim), hidden_size * 2), | Continuation line that contributes one element/argument to a larger structure. |
| 541 |             nn.ReLU(), | Continuation line that contributes one element/argument to a larger structure. |
| 542 |             nn.Dropout(dropout), | Continuation line that contributes one element/argument to a larger structure. |
| 543 |             nn.Linear(hidden_size * 2, hidden_size * 2), | Continuation line that contributes one element/argument to a larger structure. |
| 544 |             nn.ReLU(), | Continuation line that contributes one element/argument to a larger structure. |
| 545 |         ) | Closes a list/dict/function-call block opened on previous lines. |
| 546 | &nbsp; | Blank line used to visually separate logical sections. |
| 547 |         # Fusion head maps [text_state \\|\\| side_state] -&gt; token emissions for CRF. | Comment line documenting intent: Fusion head maps [text_state \\|\\| side_state] -&gt; token emissions for CRF.. |
| 548 |         self.emission_head = nn.Sequential( | Assigns computed value to variable self.emission_head for later use. |
| 549 |             nn.Linear(hidden_size * 4, hidden_size * 2), | Continuation line that contributes one element/argument to a larger structure. |
| 550 |             nn.ReLU(), | Continuation line that contributes one element/argument to a larger structure. |
| 551 |             nn.Dropout(dropout), | Continuation line that contributes one element/argument to a larger structure. |
| 552 |             nn.Linear(hidden_size * 2, num_labels), | Continuation line that contributes one element/argument to a larger structure. |
| 553 |         ) | Closes a list/dict/function-call block opened on previous lines. |
| 554 | &nbsp; | Blank line used to visually separate logical sections. |
| 555 |         self.crf = CRF(num_labels, pad_idx=None, use_gpu=torch.cuda.is_available()) | Assigns computed value to variable self.crf for later use. |
| 556 | &nbsp; | Blank line used to visually separate logical sections. |
| 557 |     def _crf_l2_penalty(self) -&gt; torch.Tensor: | Declares function _crf_l2_penalty and begins its parameter signature. |
| 558 |         penalty = torch.tensor(0.0, device=next(self.parameters()).device) | Assigns computed value to variable penalty for later use. |
| 559 |         for parameter in self.crf.parameters(): | Starts an iteration loop over a collection or iterable sequence. |
| 560 |             penalty = penalty + torch.sum(parameter ** 2) | Assigns computed value to variable penalty for later use. |
| 561 |         return penalty | Returns a value to the caller from the current function. |
| 562 | &nbsp; | Blank line used to visually separate logical sections. |
| 563 |     @staticmethod | Continuation line in an ongoing expression, literal, or control-flow block. |
| 564 |     def token_tags_to_report_labels(tag_sequences: List[List[int]]) -&gt; List[int]: | Declares function token_tags_to_report_labels and begins its parameter signature. |
| 565 |         """Collapse token predictions into one report-level duplicate decision.""" | Single-line docstring that documents behavior for the surrounding function/class/module. |
| 566 |         predictions: List[int] = [] | Continuation line in an ongoing expression, literal, or control-flow block. |
| 567 |         for tags in tag_sequences: | Starts an iteration loop over a collection or iterable sequence. |
| 568 |             if not tags: | Starts conditional logic; this block executes only when the condition is true. |
| 569 |                 predictions.append(0) | Continuation line in an ongoing expression, literal, or control-flow block. |
| 570 |                 continue | Continuation line in an ongoing expression, literal, or control-flow block. |
| 571 |             positive_ratio = float(sum(tags)) / float(len(tags)) | Assigns computed value to variable positive_ratio for later use. |
| 572 |             predictions.append(int(positive_ratio &gt;= 0.5)) | Continuation line in an ongoing expression, literal, or control-flow block. |
| 573 |         return predictions | Returns a value to the caller from the current function. |
| 574 | &nbsp; | Blank line used to visually separate logical sections. |
| 575 |     def forward( | Declares function forward and begins its parameter signature. |
| 576 |         self, | Continuation line that contributes one element/argument to a larger structure. |
| 577 |         input_ids: torch.Tensor, | Continuation line that contributes one element/argument to a larger structure. |
| 578 |         attention_mask: torch.Tensor, | Continuation line that contributes one element/argument to a larger structure. |
| 579 |         temporal_features: torch.Tensor, | Continuation line that contributes one element/argument to a larger structure. |
| 580 |         categorical_features: torch.Tensor, | Continuation line that contributes one element/argument to a larger structure. |
| 581 |         contextual_features: torch.Tensor, | Continuation line that contributes one element/argument to a larger structure. |
| 582 |         tags: Optional[torch.Tensor] = None, | Continuation line that contributes one element/argument to a larger structure. |
| 583 |     ) -&gt; Dict[str, object]: | Continuation line in an ongoing expression, literal, or control-flow block. |
| 584 |         # 1) Text branch from paper equation ht = LSTM(xt, ht-1). | Comment line documenting intent: 1) Text branch from paper equation ht = LSTM(xt, ht-1).. |
| 585 |         embeddings = self.embedding(input_ids) | Assigns computed value to variable embeddings for later use. |
| 586 |         lstm_output, _ = self.text_encoder(embeddings) | Continuation line in an ongoing expression, literal, or control-flow block. |
| 587 |         lstm_output = self.dropout(lstm_output) | Assigns computed value to variable lstm_output for later use. |
| 588 | &nbsp; | Blank line used to visually separate logical sections. |
| 589 |         # 2) Concatenate non-text features and project to match text hidden width. | Comment line documenting intent: 2) Concatenate non-text features and project to match text hidden width.. |
| 590 |         side_features = torch.cat([temporal_features, categorical_features, contextual_features], dim=-1) | Assigns computed value to variable side_features for later use. |
| 591 |         if side_features.shape[-1] == 0: | Starts conditional logic; this block executes only when the condition is true. |
| 592 |             side_features = torch.zeros((input_ids.size(0), 1), device=input_ids.device) | Assigns computed value to variable side_features for later use. |
| 593 | &nbsp; | Blank line used to visually separate logical sections. |
| 594 |         side_representation = self.side_encoder(side_features) | Assigns computed value to variable side_representation for later use. |
| 595 |         side_representation = side_representation.unsqueeze(1).expand(-1, input_ids.size(1), -1) | Assigns computed value to variable side_representation for later use. |
| 596 | &nbsp; | Blank line used to visually separate logical sections. |
| 597 |         # 3) Fuse and score each token for CRF decoding. | Comment line documenting intent: 3) Fuse and score each token for CRF decoding.. |
| 598 |         fused_representation = torch.cat([lstm_output, side_representation], dim=-1) | Assigns computed value to variable fused_representation for later use. |
| 599 |         emissions = self.emission_head(fused_representation) | Assigns computed value to variable emissions for later use. |
| 600 | &nbsp; | Blank line used to visually separate logical sections. |
| 601 |         crf_mask = cast(torch.BoolTensor, attention_mask.bool()) | Assigns computed value to variable crf_mask for later use. |
| 602 |         decoded_tags = self.crf.viterbi_decode(emissions, crf_mask) | Assigns computed value to variable decoded_tags for later use. |
| 603 | &nbsp; | Blank line used to visually separate logical sections. |
| 604 |         output: Dict[str, object] = { | Continuation line in an ongoing expression, literal, or control-flow block. |
| 605 |             "decoded_tags": decoded_tags, | Continuation line that contributes one element/argument to a larger structure. |
| 606 |             "emissions": emissions, | Continuation line that contributes one element/argument to a larger structure. |
| 607 |         } | Closes a list/dict/function-call block opened on previous lines. |
| 608 | &nbsp; | Blank line used to visually separate logical sections. |
| 609 |         if tags is not None: | Starts conditional logic; this block executes only when the condition is true. |
| 610 |             # TorchCRF returns log-likelihood; we minimize negative log-likelihood. | Comment line documenting intent: TorchCRF returns log-likelihood; we minimize negative log-likelihood.. |
| 611 |             log_likelihood = self.crf(emissions, tags.long(), crf_mask) | Assigns computed value to variable log_likelihood for later use. |
| 612 |             loss = -log_likelihood.mean() | Assigns computed value to variable loss for later use. |
| 613 | &nbsp; | Blank line used to visually separate logical sections. |
| 614 |             # Optional CRF regularization from the paper's optimizer search space. | Comment line documenting intent: Optional CRF regularization from the paper's optimizer search space.. |
| 615 |             if self.crf_regularization &gt; 0.0: | Starts conditional logic; this block executes only when the condition is true. |
| 616 |                 loss = loss + self.crf_regularization * self._crf_l2_penalty() | Assigns computed value to variable loss for later use. |
| 617 |             output["loss"] = loss | Continuation line in an ongoing expression, literal, or control-flow block. |
| 618 | &nbsp; | Blank line used to visually separate logical sections. |
| 619 |         return output | Returns a value to the caller from the current function. |
| 620 | &nbsp; | Blank line used to visually separate logical sections. |
| 621 | &nbsp; | Blank line used to visually separate logical sections. |
| 622 | @dataclass | Applies @dataclass to auto-generate constructor and utility methods for the next class. |
| 623 | class TrainConfig: | Declares class TrainConfig, creating a reusable type in the modeling pipeline. |
| 624 |     data_path: Path | Continuation line in an ongoing expression, literal, or control-flow block. |
| 625 |     model_out: Path | Continuation line in an ongoing expression, literal, or control-flow block. |
| 626 |     label_column: Optional[str] | Continuation line in an ongoing expression, literal, or control-flow block. |
| 627 |     max_vocab_size: int | Continuation line in an ongoing expression, literal, or control-flow block. |
| 628 |     max_seq_len: int | Continuation line in an ongoing expression, literal, or control-flow block. |
| 629 |     embedding_dim: int | Continuation line in an ongoing expression, literal, or control-flow block. |
| 630 |     hidden_size: int | Continuation line in an ongoing expression, literal, or control-flow block. |
| 631 |     dropout: float | Continuation line in an ongoing expression, literal, or control-flow block. |
| 632 |     learning_rate: float | Continuation line in an ongoing expression, literal, or control-flow block. |
| 633 |     crf_regularization: float | Continuation line in an ongoing expression, literal, or control-flow block. |
| 634 |     batch_size: int | Continuation line in an ongoing expression, literal, or control-flow block. |
| 635 |     epochs: int | Continuation line in an ongoing expression, literal, or control-flow block. |
| 636 |     val_size: float | Continuation line in an ongoing expression, literal, or control-flow block. |
| 637 |     seed: int | Continuation line in an ongoing expression, literal, or control-flow block. |
| 638 |     use_dragonfly: bool | Continuation line in an ongoing expression, literal, or control-flow block. |
| 639 |     dragonfly_population: int | Continuation line in an ongoing expression, literal, or control-flow block. |
| 640 |     dragonfly_iterations: int | Continuation line in an ongoing expression, literal, or control-flow block. |
| 641 |     dragonfly_inner_epochs: int | Continuation line in an ongoing expression, literal, or control-flow block. |
| 642 | &nbsp; | Blank line used to visually separate logical sections. |
| 643 | &nbsp; | Blank line used to visually separate logical sections. |
| 644 | @dataclass | Applies @dataclass to auto-generate constructor and utility methods for the next class. |
| 645 | class DataArtifacts: | Declares class DataArtifacts, creating a reusable type in the modeling pipeline. |
| 646 |     vocabulary: Vocabulary | Continuation line in an ongoing expression, literal, or control-flow block. |
| 647 |     train_dataset: BugReportDataset | Continuation line in an ongoing expression, literal, or control-flow block. |
| 648 |     val_dataset: BugReportDataset | Continuation line in an ongoing expression, literal, or control-flow block. |
| 649 |     train_loader: DataLoader | Continuation line in an ongoing expression, literal, or control-flow block. |
| 650 |     val_loader: DataLoader | Continuation line in an ongoing expression, literal, or control-flow block. |
| 651 | &nbsp; | Blank line used to visually separate logical sections. |
| 652 | &nbsp; | Blank line used to visually separate logical sections. |
| 653 | @dataclass | Applies @dataclass to auto-generate constructor and utility methods for the next class. |
| 654 | class HyperCandidate: | Declares class HyperCandidate, creating a reusable type in the modeling pipeline. |
| 655 |     learning_rate: float | Continuation line in an ongoing expression, literal, or control-flow block. |
| 656 |     hidden_size: int | Continuation line in an ongoing expression, literal, or control-flow block. |
| 657 |     dropout: float | Continuation line in an ongoing expression, literal, or control-flow block. |
| 658 |     crf_regularization: float | Continuation line in an ongoing expression, literal, or control-flow block. |
| 659 | &nbsp; | Blank line used to visually separate logical sections. |
| 660 | &nbsp; | Blank line used to visually separate logical sections. |
| 661 | class DragonflyHyperOptimizer: | Declares class DragonflyHyperOptimizer, creating a reusable type in the modeling pipeline. |
| 662 |     """ | Start of a multi-line docstring block. |
| 663 |     Lightweight Dragonfly-inspired optimizer for ADLM hyperparameters. | Continuation line in an ongoing expression, literal, or control-flow block. |
| 664 |     This mirrors the paper's idea: optimize LR, hidden units, dropout, CRF regularization. | Continuation line in an ongoing expression, literal, or control-flow block. |
| 665 |     """ | Start of a multi-line docstring block. |
| 666 | &nbsp; | Blank line used to visually separate logical sections. |
| 667 |     def __init__(self, population: int, iterations: int, seed: int) -&gt; None: | Declares function __init__ and begins its parameter signature. |
| 668 |         self.population = population | Assigns computed value to variable self.population for later use. |
| 669 |         self.iterations = iterations | Assigns computed value to variable self.iterations for later use. |
| 670 |         self.random = np.random.default_rng(seed) | Assigns computed value to variable self.random for later use. |
| 671 |         self.bounds = { | Assigns computed value to variable self.bounds for later use. |
| 672 |             "learning_rate": (1e-4, 5e-3), | Continuation line that contributes one element/argument to a larger structure. |
| 673 |             "hidden_size": (48.0, 256.0), | Continuation line that contributes one element/argument to a larger structure. |
| 674 |             "dropout": (0.1, 0.6), | Continuation line that contributes one element/argument to a larger structure. |
| 675 |             "crf_regularization": (1e-6, 5e-3), | Continuation line that contributes one element/argument to a larger structure. |
| 676 |         } | Closes a list/dict/function-call block opened on previous lines. |
| 677 | &nbsp; | Blank line used to visually separate logical sections. |
| 678 |     def _clip(self, vector: np.ndarray) -&gt; np.ndarray: | Declares function _clip and begins its parameter signature. |
| 679 |         clipped = vector.copy() | Assigns computed value to variable clipped for later use. |
| 680 |         keys = list(self.bounds.keys()) | Assigns computed value to variable keys for later use. |
| 681 |         for idx, key in enumerate(keys): | Starts an iteration loop over a collection or iterable sequence. |
| 682 |             lo, hi = self.bounds[key] | Continuation line in an ongoing expression, literal, or control-flow block. |
| 683 |             clipped[idx] = np.clip(clipped[idx], lo, hi) | Continuation line in an ongoing expression, literal, or control-flow block. |
| 684 |         return clipped | Returns a value to the caller from the current function. |
| 685 | &nbsp; | Blank line used to visually separate logical sections. |
| 686 |     def _to_candidate(self, vector: np.ndarray) -&gt; HyperCandidate: | Declares function _to_candidate and begins its parameter signature. |
| 687 |         return HyperCandidate( | Returns a value to the caller from the current function. |
| 688 |             learning_rate=float(vector[0]), | Assigns computed value to variable learning_rate for later use. |
| 689 |             hidden_size=int(round(float(vector[1]) / 8.0) * 8), | Assigns computed value to variable hidden_size for later use. |
| 690 |             dropout=float(vector[2]), | Assigns computed value to variable dropout for later use. |
| 691 |             crf_regularization=float(vector[3]), | Assigns computed value to variable crf_regularization for later use. |
| 692 |         ) | Closes a list/dict/function-call block opened on previous lines. |
| 693 | &nbsp; | Blank line used to visually separate logical sections. |
| 694 |     def _sample_population(self) -&gt; np.ndarray: | Declares function _sample_population and begins its parameter signature. |
| 695 |         vectors = [] | Assigns computed value to variable vectors for later use. |
| 696 |         for _ in range(self.population): | Starts an iteration loop over a collection or iterable sequence. |
| 697 |             vectors.append( | Continuation line in an ongoing expression, literal, or control-flow block. |
| 698 |                 np.array( | Continuation line in an ongoing expression, literal, or control-flow block. |
| 699 |                     [ | Continuation line in an ongoing expression, literal, or control-flow block. |
| 700 |                         self.random.uniform(*self.bounds["learning_rate"]), | Continuation line that contributes one element/argument to a larger structure. |
| 701 |                         self.random.uniform(*self.bounds["hidden_size"]), | Continuation line that contributes one element/argument to a larger structure. |
| 702 |                         self.random.uniform(*self.bounds["dropout"]), | Continuation line that contributes one element/argument to a larger structure. |
| 703 |                         self.random.uniform(*self.bounds["crf_regularization"]), | Continuation line that contributes one element/argument to a larger structure. |
| 704 |                     ], | Closes a list/dict/function-call block opened on previous lines. |
| 705 |                     dtype=np.float64, | Assigns computed value to variable dtype for later use. |
| 706 |                 ) | Closes a list/dict/function-call block opened on previous lines. |
| 707 |             ) | Closes a list/dict/function-call block opened on previous lines. |
| 708 |         return np.stack(vectors) | Returns a value to the caller from the current function. |
| 709 | &nbsp; | Blank line used to visually separate logical sections. |
| 710 |     def optimize(self, objective: Callable[[HyperCandidate], float]) -&gt; HyperCandidate: | Declares function optimize and begins its parameter signature. |
| 711 |         population = self._sample_population() | Assigns computed value to variable population for later use. |
| 712 |         delta = np.zeros_like(population) | Assigns computed value to variable delta for later use. |
| 713 | &nbsp; | Blank line used to visually separate logical sections. |
| 714 |         fitness = np.array([objective(self._to_candidate(v)) for v in population], dtype=np.float64) | Assigns computed value to variable fitness for later use. |
| 715 | &nbsp; | Blank line used to visually separate logical sections. |
| 716 |         for _ in range(self.iterations): | Starts an iteration loop over a collection or iterable sequence. |
| 717 |             best_index = int(np.argmax(fitness)) | Assigns computed value to variable best_index for later use. |
| 718 |             worst_index = int(np.argmin(fitness)) | Assigns computed value to variable worst_index for later use. |
| 719 |             best_position = population[best_index] | Assigns computed value to variable best_position for later use. |
| 720 |             worst_position = population[worst_index] | Assigns computed value to variable worst_position for later use. |
| 721 |             mean_position = np.mean(population, axis=0) | Assigns computed value to variable mean_position for later use. |
| 722 |             mean_velocity = np.mean(delta, axis=0) | Assigns computed value to variable mean_velocity for later use. |
| 723 | &nbsp; | Blank line used to visually separate logical sections. |
| 724 |             for i in range(self.population): | Starts an iteration loop over a collection or iterable sequence. |
| 725 |                 # Simplified Dragonfly forces: separation, alignment, cohesion, food, enemy. | Comment line documenting intent: Simplified Dragonfly forces: separation, alignment, cohesion, food, enemy.. |
| 726 |                 separation = -(population[i] - mean_position) | Assigns computed value to variable separation for later use. |
| 727 |                 alignment = mean_velocity | Assigns computed value to variable alignment for later use. |
| 728 |                 cohesion = mean_position - population[i] | Assigns computed value to variable cohesion for later use. |
| 729 |                 food = best_position - population[i] | Assigns computed value to variable food for later use. |
| 730 |                 enemy = population[i] - worst_position | Assigns computed value to variable enemy for later use. |
| 731 |                 noise = self.random.normal(0.0, 0.01, size=population.shape[1]) | Assigns computed value to variable noise for later use. |
| 732 | &nbsp; | Blank line used to visually separate logical sections. |
| 733 |                 delta[i] = ( | Continuation line in an ongoing expression, literal, or control-flow block. |
| 734 |                     0.35 * delta[i] | Continuation line in an ongoing expression, literal, or control-flow block. |
| 735 |                     + 0.10 * separation | Continuation line in an ongoing expression, literal, or control-flow block. |
| 736 |                     + 0.10 * alignment | Continuation line in an ongoing expression, literal, or control-flow block. |
| 737 |                     + 0.10 * cohesion | Continuation line in an ongoing expression, literal, or control-flow block. |
| 738 |                     + 0.40 * food | Continuation line in an ongoing expression, literal, or control-flow block. |
| 739 |                     + 0.20 * enemy | Continuation line in an ongoing expression, literal, or control-flow block. |
| 740 |                     + noise | Continuation line in an ongoing expression, literal, or control-flow block. |
| 741 |                 ) | Closes a list/dict/function-call block opened on previous lines. |
| 742 |                 population[i] = self._clip(population[i] + delta[i]) | Continuation line in an ongoing expression, literal, or control-flow block. |
| 743 | &nbsp; | Blank line used to visually separate logical sections. |
| 744 |             fitness = np.array([objective(self._to_candidate(v)) for v in population], dtype=np.float64) | Assigns computed value to variable fitness for later use. |
| 745 | &nbsp; | Blank line used to visually separate logical sections. |
| 746 |         best_index = int(np.argmax(fitness)) | Assigns computed value to variable best_index for later use. |
| 747 |         return self._to_candidate(population[best_index]) | Returns a value to the caller from the current function. |
| 748 | &nbsp; | Blank line used to visually separate logical sections. |
| 749 | &nbsp; | Blank line used to visually separate logical sections. |
| 750 | def compute_metrics(targets: List[int], predictions: List[int]) -&gt; Dict[str, float]: | Declares function compute_metrics and begins its parameter signature. |
| 751 |     accuracy = accuracy_score(targets, predictions) | Assigns computed value to variable accuracy for later use. |
| 752 |     precision, recall, f1, _ = precision_recall_fscore_support( | Continuation line in an ongoing expression, literal, or control-flow block. |
| 753 |         targets, | Continuation line that contributes one element/argument to a larger structure. |
| 754 |         predictions, | Continuation line that contributes one element/argument to a larger structure. |
| 755 |         average="binary", | Assigns computed value to variable average for later use. |
| 756 |         zero_division=0, | Assigns computed value to variable zero_division for later use. |
| 757 |     ) | Closes a list/dict/function-call block opened on previous lines. |
| 758 |     kappa = cohen_kappa_score(targets, predictions) | Assigns computed value to variable kappa for later use. |
| 759 |     return { | Returns a value to the caller from the current function. |
| 760 |         "accuracy": float(accuracy), | Continuation line that contributes one element/argument to a larger structure. |
| 761 |         "precision": float(precision), | Continuation line that contributes one element/argument to a larger structure. |
| 762 |         "recall": float(recall), | Continuation line that contributes one element/argument to a larger structure. |
| 763 |         "f1": float(f1), | Continuation line that contributes one element/argument to a larger structure. |
| 764 |         "kappa": float(kappa), | Continuation line that contributes one element/argument to a larger structure. |
| 765 |     } | Closes a list/dict/function-call block opened on previous lines. |
| 766 | &nbsp; | Blank line used to visually separate logical sections. |
| 767 | &nbsp; | Blank line used to visually separate logical sections. |
| 768 | def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -&gt; Dict[str, torch.Tensor]: | Declares function move_batch_to_device and begins its parameter signature. |
| 769 |     moved: Dict[str, torch.Tensor] = {} | Continuation line in an ongoing expression, literal, or control-flow block. |
| 770 |     for key, value in batch.items(): | Starts an iteration loop over a collection or iterable sequence. |
| 771 |         moved[key] = value.to(device) | Continuation line in an ongoing expression, literal, or control-flow block. |
| 772 |     return moved | Returns a value to the caller from the current function. |
| 773 | &nbsp; | Blank line used to visually separate logical sections. |
| 774 | &nbsp; | Blank line used to visually separate logical sections. |
| 775 | def train_one_epoch( | Declares function train_one_epoch and begins its parameter signature. |
| 776 |     model: ADLMBiLSTMCRF, | Continuation line that contributes one element/argument to a larger structure. |
| 777 |     dataloader: DataLoader, | Continuation line that contributes one element/argument to a larger structure. |
| 778 |     optimizer: torch.optim.Optimizer, | Continuation line that contributes one element/argument to a larger structure. |
| 779 |     device: torch.device, | Continuation line that contributes one element/argument to a larger structure. |
| 780 | ) -&gt; float: | Continuation line in an ongoing expression, literal, or control-flow block. |
| 781 |     model.train() | Continuation line in an ongoing expression, literal, or control-flow block. |
| 782 |     losses: List[float] = [] | Continuation line in an ongoing expression, literal, or control-flow block. |
| 783 | &nbsp; | Blank line used to visually separate logical sections. |
| 784 |     for batch in dataloader: | Starts an iteration loop over a collection or iterable sequence. |
| 785 |         batch = move_batch_to_device(batch, device) | Assigns computed value to variable batch for later use. |
| 786 |         optimizer.zero_grad() | Continuation line in an ongoing expression, literal, or control-flow block. |
| 787 | &nbsp; | Blank line used to visually separate logical sections. |
| 788 |         output = model( | Assigns computed value to variable output for later use. |
| 789 |             input_ids=batch["input_ids"], | Assigns computed value to variable input_ids for later use. |
| 790 |             attention_mask=batch["attention_mask"], | Assigns computed value to variable attention_mask for later use. |
| 791 |             temporal_features=batch["temporal_features"], | Assigns computed value to variable temporal_features for later use. |
| 792 |             categorical_features=batch["categorical_features"], | Assigns computed value to variable categorical_features for later use. |
| 793 |             contextual_features=batch["contextual_features"], | Assigns computed value to variable contextual_features for later use. |
| 794 |             tags=batch["tag_ids"], | Assigns computed value to variable tags for later use. |
| 795 |         ) | Closes a list/dict/function-call block opened on previous lines. |
| 796 |         loss = output["loss"] | Assigns computed value to variable loss for later use. |
| 797 |         loss.backward() | Continuation line in an ongoing expression, literal, or control-flow block. |
| 798 |         nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) | Continuation line in an ongoing expression, literal, or control-flow block. |
| 799 |         optimizer.step() | Continuation line in an ongoing expression, literal, or control-flow block. |
| 800 | &nbsp; | Blank line used to visually separate logical sections. |
| 801 |         losses.append(float(loss.detach().cpu().item())) | Continuation line in an ongoing expression, literal, or control-flow block. |
| 802 | &nbsp; | Blank line used to visually separate logical sections. |
| 803 |     return float(np.mean(losses)) if losses else 0.0 | Returns a value to the caller from the current function. |
| 804 | &nbsp; | Blank line used to visually separate logical sections. |
| 805 | &nbsp; | Blank line used to visually separate logical sections. |
| 806 | def evaluate( | Declares function evaluate and begins its parameter signature. |
| 807 |     model: ADLMBiLSTMCRF, | Continuation line that contributes one element/argument to a larger structure. |
| 808 |     dataloader: DataLoader, | Continuation line that contributes one element/argument to a larger structure. |
| 809 |     device: torch.device, | Continuation line that contributes one element/argument to a larger structure. |
| 810 | ) -&gt; Dict[str, float]: | Continuation line in an ongoing expression, literal, or control-flow block. |
| 811 |     model.eval() | Continuation line in an ongoing expression, literal, or control-flow block. |
| 812 |     losses: List[float] = [] | Continuation line in an ongoing expression, literal, or control-flow block. |
| 813 |     targets: List[int] = [] | Continuation line in an ongoing expression, literal, or control-flow block. |
| 814 |     predictions: List[int] = [] | Continuation line in an ongoing expression, literal, or control-flow block. |
| 815 | &nbsp; | Blank line used to visually separate logical sections. |
| 816 |     with torch.no_grad(): | Starts a context-manager block for safe setup/cleanup of resources. |
| 817 |         for batch in dataloader: | Starts an iteration loop over a collection or iterable sequence. |
| 818 |             batch = move_batch_to_device(batch, device) | Assigns computed value to variable batch for later use. |
| 819 |             output = model( | Assigns computed value to variable output for later use. |
| 820 |                 input_ids=batch["input_ids"], | Assigns computed value to variable input_ids for later use. |
| 821 |                 attention_mask=batch["attention_mask"], | Assigns computed value to variable attention_mask for later use. |
| 822 |                 temporal_features=batch["temporal_features"], | Assigns computed value to variable temporal_features for later use. |
| 823 |                 categorical_features=batch["categorical_features"], | Assigns computed value to variable categorical_features for later use. |
| 824 |                 contextual_features=batch["contextual_features"], | Assigns computed value to variable contextual_features for later use. |
| 825 |                 tags=batch["tag_ids"], | Assigns computed value to variable tags for later use. |
| 826 |             ) | Closes a list/dict/function-call block opened on previous lines. |
| 827 | &nbsp; | Blank line used to visually separate logical sections. |
| 828 |             if "loss" in output: | Starts conditional logic; this block executes only when the condition is true. |
| 829 |                 losses.append(float(output["loss"].detach().cpu().item())) | Continuation line in an ongoing expression, literal, or control-flow block. |
| 830 | &nbsp; | Blank line used to visually separate logical sections. |
| 831 |             decoded = output["decoded_tags"] | Assigns computed value to variable decoded for later use. |
| 832 |             batch_predictions = ADLMBiLSTMCRF.token_tags_to_report_labels(decoded) | Assigns computed value to variable batch_predictions for later use. |
| 833 |             batch_targets = batch["label"].detach().cpu().tolist() | Assigns computed value to variable batch_targets for later use. |
| 834 | &nbsp; | Blank line used to visually separate logical sections. |
| 835 |             predictions.extend(batch_predictions) | Continuation line in an ongoing expression, literal, or control-flow block. |
| 836 |             targets.extend(batch_targets) | Continuation line in an ongoing expression, literal, or control-flow block. |
| 837 | &nbsp; | Blank line used to visually separate logical sections. |
| 838 |     metrics = compute_metrics(targets, predictions) | Assigns computed value to variable metrics for later use. |
| 839 |     metrics["loss"] = float(np.mean(losses)) if losses else 0.0 | Continuation line in an ongoing expression, literal, or control-flow block. |
| 840 |     return metrics | Returns a value to the caller from the current function. |
| 841 | &nbsp; | Blank line used to visually separate logical sections. |
| 842 | &nbsp; | Blank line used to visually separate logical sections. |
| 843 | def split_examples(examples: Sequence[BugExample], val_size: float, seed: int) -&gt; List[List[BugExample]]: | Declares function split_examples and begins its parameter signature. |
| 844 |     labels = [example.label for example in examples] | Assigns computed value to variable labels for later use. |
| 845 | &nbsp; | Blank line used to visually separate logical sections. |
| 846 |     # Stratify only when both classes are present with enough samples. | Comment line documenting intent: Stratify only when both classes are present with enough samples.. |
| 847 |     stratify = None | Assigns computed value to variable stratify for later use. |
| 848 |     label_set = set(labels) | Assigns computed value to variable label_set for later use. |
| 849 |     if len(label_set) &gt; 1: | Starts conditional logic; this block executes only when the condition is true. |
| 850 |         counts = Counter(labels) | Assigns computed value to variable counts for later use. |
| 851 |         if min(counts.values()) &gt;= 2: | Starts conditional logic; this block executes only when the condition is true. |
| 852 |             stratify = labels | Assigns computed value to variable stratify for later use. |
| 853 | &nbsp; | Blank line used to visually separate logical sections. |
| 854 |     train_examples, val_examples = train_test_split( | Continuation line in an ongoing expression, literal, or control-flow block. |
| 855 |         list(examples), | Function or method call that triggers computation or side effects. |
| 856 |         test_size=val_size, | Assigns computed value to variable test_size for later use. |
| 857 |         random_state=seed, | Assigns computed value to variable random_state for later use. |
| 858 |         stratify=stratify, | Assigns computed value to variable stratify for later use. |
| 859 |     ) | Closes a list/dict/function-call block opened on previous lines. |
| 860 |     return [train_examples, val_examples] | Returns a value to the caller from the current function. |
| 861 | &nbsp; | Blank line used to visually separate logical sections. |
| 862 | &nbsp; | Blank line used to visually separate logical sections. |
| 863 | def build_dataloaders( | Declares function build_dataloaders and begins its parameter signature. |
| 864 |     train_examples: Sequence[BugExample], | Continuation line that contributes one element/argument to a larger structure. |
| 865 |     val_examples: Sequence[BugExample], | Continuation line that contributes one element/argument to a larger structure. |
| 866 |     max_vocab_size: int, | Continuation line that contributes one element/argument to a larger structure. |
| 867 |     max_seq_len: int, | Continuation line that contributes one element/argument to a larger structure. |
| 868 |     batch_size: int, | Continuation line that contributes one element/argument to a larger structure. |
| 869 | ) -&gt; DataArtifacts: | Continuation line in an ongoing expression, literal, or control-flow block. |
| 870 |     vocabulary = Vocabulary() | Assigns computed value to variable vocabulary for later use. |
| 871 |     vocabulary.build([example.text for example in train_examples], max_size=max_vocab_size) | Continuation line in an ongoing expression, literal, or control-flow block. |
| 872 | &nbsp; | Blank line used to visually separate logical sections. |
| 873 |     train_dataset = BugReportDataset(train_examples, vocabulary=vocabulary, max_seq_len=max_seq_len) | Assigns computed value to variable train_dataset for later use. |
| 874 |     val_dataset = BugReportDataset(val_examples, vocabulary=vocabulary, max_seq_len=max_seq_len) | Assigns computed value to variable val_dataset for later use. |
| 875 | &nbsp; | Blank line used to visually separate logical sections. |
| 876 |     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) | Assigns computed value to variable train_loader for later use. |
| 877 |     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) | Assigns computed value to variable val_loader for later use. |
| 878 | &nbsp; | Blank line used to visually separate logical sections. |
| 879 |     return DataArtifacts( | Returns a value to the caller from the current function. |
| 880 |         vocabulary=vocabulary, | Assigns computed value to variable vocabulary for later use. |
| 881 |         train_dataset=train_dataset, | Assigns computed value to variable train_dataset for later use. |
| 882 |         val_dataset=val_dataset, | Assigns computed value to variable val_dataset for later use. |
| 883 |         train_loader=train_loader, | Assigns computed value to variable train_loader for later use. |
| 884 |         val_loader=val_loader, | Assigns computed value to variable val_loader for later use. |
| 885 |     ) | Closes a list/dict/function-call block opened on previous lines. |
| 886 | &nbsp; | Blank line used to visually separate logical sections. |
| 887 | &nbsp; | Blank line used to visually separate logical sections. |
| 888 | def run_training(config: TrainConfig) -&gt; None: | Declares function run_training and begins its parameter signature. |
| 889 |     set_seed(config.seed) | Function or method call that triggers computation or side effects. |
| 890 | &nbsp; | Blank line used to visually separate logical sections. |
| 891 |     dataframe = load_modeling_dataframe(config.data_path) | Assigns computed value to variable dataframe for later use. |
| 892 |     label_column = detect_label_column(dataframe, config.label_column) | Assigns computed value to variable label_column for later use. |
| 893 |     examples = build_examples(dataframe, label_column=label_column) | Assigns computed value to variable examples for later use. |
| 894 |     if len(examples) &lt; 10: | Starts conditional logic; this block executes only when the condition is true. |
| 895 |         raise ValueError("Not enough valid training examples found in the dataset.") | Raises an exception to signal an error or invalid state. |
| 896 | &nbsp; | Blank line used to visually separate logical sections. |
| 897 |     train_examples, val_examples = split_examples(examples, val_size=config.val_size, seed=config.seed) | Continuation line in an ongoing expression, literal, or control-flow block. |
| 898 |     artifacts = build_dataloaders( | Assigns computed value to variable artifacts for later use. |
| 899 |         train_examples=train_examples, | Assigns computed value to variable train_examples for later use. |
| 900 |         val_examples=val_examples, | Assigns computed value to variable val_examples for later use. |
| 901 |         max_vocab_size=config.max_vocab_size, | Assigns computed value to variable max_vocab_size for later use. |
| 902 |         max_seq_len=config.max_seq_len, | Assigns computed value to variable max_seq_len for later use. |
| 903 |         batch_size=config.batch_size, | Assigns computed value to variable batch_size for later use. |
| 904 |     ) | Closes a list/dict/function-call block opened on previous lines. |
| 905 | &nbsp; | Blank line used to visually separate logical sections. |
| 906 |     train_dataset = artifacts.train_dataset | Assigns computed value to variable train_dataset for later use. |
| 907 |     train_loader = artifacts.train_loader | Assigns computed value to variable train_loader for later use. |
| 908 |     val_loader = artifacts.val_loader | Assigns computed value to variable val_loader for later use. |
| 909 |     vocabulary = artifacts.vocabulary | Assigns computed value to variable vocabulary for later use. |
| 910 | &nbsp; | Blank line used to visually separate logical sections. |
| 911 |     device = torch.device("cuda" if torch.cuda.is_available() else "cpu") | Assigns computed value to variable device for later use. |
| 912 | &nbsp; | Blank line used to visually separate logical sections. |
| 913 |     def build_model(candidate: Optional[HyperCandidate] = None) -&gt; ADLMBiLSTMCRF: | Declares function build_model and begins its parameter signature. |
| 914 |         hidden_size = candidate.hidden_size if candidate else config.hidden_size | Assigns computed value to variable hidden_size for later use. |
| 915 |         dropout = candidate.dropout if candidate else config.dropout | Assigns computed value to variable dropout for later use. |
| 916 |         crf_reg = candidate.crf_regularization if candidate else config.crf_regularization | Assigns computed value to variable crf_reg for later use. |
| 917 | &nbsp; | Blank line used to visually separate logical sections. |
| 918 |         model = ADLMBiLSTMCRF( | Assigns computed value to variable model for later use. |
| 919 |             vocab_size=vocabulary.size, | Assigns computed value to variable vocab_size for later use. |
| 920 |             embedding_dim=config.embedding_dim, | Assigns computed value to variable embedding_dim for later use. |
| 921 |             hidden_size=hidden_size, | Assigns computed value to variable hidden_size for later use. |
| 922 |             dropout=dropout, | Assigns computed value to variable dropout for later use. |
| 923 |             temporal_dim=train_dataset.temporal_dim, | Assigns computed value to variable temporal_dim for later use. |
| 924 |             categorical_dim=train_dataset.categorical_dim, | Assigns computed value to variable categorical_dim for later use. |
| 925 |             contextual_dim=train_dataset.contextual_dim, | Assigns computed value to variable contextual_dim for later use. |
| 926 |             crf_regularization=crf_reg, | Assigns computed value to variable crf_regularization for later use. |
| 927 |         ) | Closes a list/dict/function-call block opened on previous lines. |
| 928 |         return model.to(device) | Returns a value to the caller from the current function. |
| 929 | &nbsp; | Blank line used to visually separate logical sections. |
| 930 |     chosen_lr = config.learning_rate | Assigns computed value to variable chosen_lr for later use. |
| 931 |     chosen_hidden = config.hidden_size | Assigns computed value to variable chosen_hidden for later use. |
| 932 |     chosen_dropout = config.dropout | Assigns computed value to variable chosen_dropout for later use. |
| 933 |     chosen_crf_reg = config.crf_regularization | Assigns computed value to variable chosen_crf_reg for later use. |
| 934 | &nbsp; | Blank line used to visually separate logical sections. |
| 935 |     final_model: Optional[ADLMBiLSTMCRF] = None | Continuation line in an ongoing expression, literal, or control-flow block. |
| 936 |     best_state: Optional[Dict[str, torch.Tensor]] = None | Continuation line in an ongoing expression, literal, or control-flow block. |
| 937 |     best_f1 = -1.0 | Assigns computed value to variable best_f1 for later use. |
| 938 |     best_metrics: Dict[str, float] = {} | Continuation line in an ongoing expression, literal, or control-flow block. |
| 939 |     training_status = "completed" | Assigns computed value to variable training_status for later use. |
| 940 |     last_completed_epoch = 0 | Assigns computed value to variable last_completed_epoch for later use. |
| 941 | &nbsp; | Blank line used to visually separate logical sections. |
| 942 |     try: | Begins try-block for guarded execution that may raise exceptions. |
| 943 |         if config.use_dragonfly: | Starts conditional logic; this block executes only when the condition is true. |
| 944 |             def objective(candidate: HyperCandidate) -&gt; float: | Declares function objective and begins its parameter signature. |
| 945 |                 model = build_model(candidate) | Assigns computed value to variable model for later use. |
| 946 |                 optimizer = torch.optim.Adam(model.parameters(), lr=candidate.learning_rate) | Assigns computed value to variable optimizer for later use. |
| 947 | &nbsp; | Blank line used to visually separate logical sections. |
| 948 |                 # Quick inner-loop training during hyperparameter search. | Comment line documenting intent: Quick inner-loop training during hyperparameter search.. |
| 949 |                 for _ in range(config.dragonfly_inner_epochs): | Starts an iteration loop over a collection or iterable sequence. |
| 950 |                     train_one_epoch(model, train_loader, optimizer, device) | Function or method call that triggers computation or side effects. |
| 951 | &nbsp; | Blank line used to visually separate logical sections. |
| 952 |                 metrics = evaluate(model, val_loader, device) | Assigns computed value to variable metrics for later use. |
| 953 |                 return metrics["f1"] | Returns a value to the caller from the current function. |
| 954 | &nbsp; | Blank line used to visually separate logical sections. |
| 955 |             optimizer_do = DragonflyHyperOptimizer( | Assigns computed value to variable optimizer_do for later use. |
| 956 |                 population=config.dragonfly_population, | Assigns computed value to variable population for later use. |
| 957 |                 iterations=config.dragonfly_iterations, | Assigns computed value to variable iterations for later use. |
| 958 |                 seed=config.seed, | Assigns computed value to variable seed for later use. |
| 959 |             ) | Closes a list/dict/function-call block opened on previous lines. |
| 960 |             best_candidate = optimizer_do.optimize(objective) | Assigns computed value to variable best_candidate for later use. |
| 961 |             chosen_lr = best_candidate.learning_rate | Assigns computed value to variable chosen_lr for later use. |
| 962 |             chosen_hidden = best_candidate.hidden_size | Assigns computed value to variable chosen_hidden for later use. |
| 963 |             chosen_dropout = best_candidate.dropout | Assigns computed value to variable chosen_dropout for later use. |
| 964 |             chosen_crf_reg = best_candidate.crf_regularization | Assigns computed value to variable chosen_crf_reg for later use. |
| 965 |             print( | Prints runtime information to the console for monitoring/debugging. |
| 966 |                 "[Dragonfly] best candidate:", | Continuation line that contributes one element/argument to a larger structure. |
| 967 |                 { | Continuation line in an ongoing expression, literal, or control-flow block. |
| 968 |                     "learning_rate": chosen_lr, | Continuation line that contributes one element/argument to a larger structure. |
| 969 |                     "hidden_size": chosen_hidden, | Continuation line that contributes one element/argument to a larger structure. |
| 970 |                     "dropout": chosen_dropout, | Continuation line that contributes one element/argument to a larger structure. |
| 971 |                     "crf_regularization": chosen_crf_reg, | Continuation line that contributes one element/argument to a larger structure. |
| 972 |                 }, | Closes a list/dict/function-call block opened on previous lines. |
| 973 |             ) | Closes a list/dict/function-call block opened on previous lines. |
| 974 | &nbsp; | Blank line used to visually separate logical sections. |
| 975 |         final_model = ADLMBiLSTMCRF( | Assigns computed value to variable final_model for later use. |
| 976 |             vocab_size=vocabulary.size, | Assigns computed value to variable vocab_size for later use. |
| 977 |             embedding_dim=config.embedding_dim, | Assigns computed value to variable embedding_dim for later use. |
| 978 |             hidden_size=chosen_hidden, | Assigns computed value to variable hidden_size for later use. |
| 979 |             dropout=chosen_dropout, | Assigns computed value to variable dropout for later use. |
| 980 |             temporal_dim=train_dataset.temporal_dim, | Assigns computed value to variable temporal_dim for later use. |
| 981 |             categorical_dim=train_dataset.categorical_dim, | Assigns computed value to variable categorical_dim for later use. |
| 982 |             contextual_dim=train_dataset.contextual_dim, | Assigns computed value to variable contextual_dim for later use. |
| 983 |             crf_regularization=chosen_crf_reg, | Assigns computed value to variable crf_regularization for later use. |
| 984 |         ).to(device) | Continuation line in an ongoing expression, literal, or control-flow block. |
| 985 | &nbsp; | Blank line used to visually separate logical sections. |
| 986 |         final_optimizer = torch.optim.Adam(final_model.parameters(), lr=chosen_lr) | Assigns computed value to variable final_optimizer for later use. |
| 987 | &nbsp; | Blank line used to visually separate logical sections. |
| 988 |         for epoch in range(1, config.epochs + 1): | Starts an iteration loop over a collection or iterable sequence. |
| 989 |             train_loss = train_one_epoch(final_model, train_loader, final_optimizer, device) | Assigns computed value to variable train_loss for later use. |
| 990 |             val_metrics = evaluate(final_model, val_loader, device) | Assigns computed value to variable val_metrics for later use. |
| 991 |             last_completed_epoch = epoch | Assigns computed value to variable last_completed_epoch for later use. |
| 992 | &nbsp; | Blank line used to visually separate logical sections. |
| 993 |             print( | Prints runtime information to the console for monitoring/debugging. |
| 994 |                 f"Epoch {epoch:02d} \\| " | Continuation line in an ongoing expression, literal, or control-flow block. |
| 995 |                 f"train_loss={train_loss:.4f} \\| " | Continuation line in an ongoing expression, literal, or control-flow block. |
| 996 |                 f"val_loss={val_metrics['loss']:.4f} \\| " | Continuation line in an ongoing expression, literal, or control-flow block. |
| 997 |                 f"val_acc={val_metrics['accuracy']:.4f} \\| " | Continuation line in an ongoing expression, literal, or control-flow block. |
| 998 |                 f"val_precision={val_metrics['precision']:.4f} \\| " | Continuation line in an ongoing expression, literal, or control-flow block. |
| 999 |                 f"val_recall={val_metrics['recall']:.4f} \\| " | Continuation line in an ongoing expression, literal, or control-flow block. |
| 1000 |                 f"val_f1={val_metrics['f1']:.4f} \\| " | Continuation line in an ongoing expression, literal, or control-flow block. |
| 1001 |                 f"val_kappa={val_metrics['kappa']:.4f}" | Continuation line in an ongoing expression, literal, or control-flow block. |
| 1002 |             ) | Closes a list/dict/function-call block opened on previous lines. |
| 1003 | &nbsp; | Blank line used to visually separate logical sections. |
| 1004 |             if val_metrics["f1"] &gt; best_f1: | Starts conditional logic; this block executes only when the condition is true. |
| 1005 |                 best_f1 = val_metrics["f1"] | Assigns computed value to variable best_f1 for later use. |
| 1006 |                 best_metrics = val_metrics | Assigns computed value to variable best_metrics for later use. |
| 1007 |                 best_state = {k: v.detach().cpu() for k, v in final_model.state_dict().items()} | Assigns computed value to variable best_state for later use. |
| 1008 | &nbsp; | Blank line used to visually separate logical sections. |
| 1009 |     except KeyboardInterrupt: | Catches and handles exceptions from the preceding try block. |
| 1010 |         training_status = "interrupted" | Assigns computed value to variable training_status for later use. |
| 1011 |         print("\nKeyboardInterrupt received. Saving checkpoint from current training state...") | Prints runtime information to the console for monitoring/debugging. |
| 1012 | &nbsp; | Blank line used to visually separate logical sections. |
| 1013 |         # If interruption happened very early (e.g., during hyperparameter search), | Comment line documenting intent: If interruption happened very early (e.g., during hyperparameter search),. |
| 1014 |         # still initialize a model so a valid checkpoint is always produced. | Comment line documenting intent: still initialize a model so a valid checkpoint is always produced.. |
| 1015 |         if final_model is None: | Starts conditional logic; this block executes only when the condition is true. |
| 1016 |             final_model = ADLMBiLSTMCRF( | Assigns computed value to variable final_model for later use. |
| 1017 |                 vocab_size=vocabulary.size, | Assigns computed value to variable vocab_size for later use. |
| 1018 |                 embedding_dim=config.embedding_dim, | Assigns computed value to variable embedding_dim for later use. |
| 1019 |                 hidden_size=chosen_hidden, | Assigns computed value to variable hidden_size for later use. |
| 1020 |                 dropout=chosen_dropout, | Assigns computed value to variable dropout for later use. |
| 1021 |                 temporal_dim=train_dataset.temporal_dim, | Assigns computed value to variable temporal_dim for later use. |
| 1022 |                 categorical_dim=train_dataset.categorical_dim, | Assigns computed value to variable categorical_dim for later use. |
| 1023 |                 contextual_dim=train_dataset.contextual_dim, | Assigns computed value to variable contextual_dim for later use. |
| 1024 |                 crf_regularization=chosen_crf_reg, | Assigns computed value to variable crf_regularization for later use. |
| 1025 |             ).to(device) | Continuation line in an ongoing expression, literal, or control-flow block. |
| 1026 | &nbsp; | Blank line used to visually separate logical sections. |
| 1027 |     if final_model is None: | Starts conditional logic; this block executes only when the condition is true. |
| 1028 |         raise RuntimeError("Unable to initialize model for checkpoint saving.") | Raises an exception to signal an error or invalid state. |
| 1029 | &nbsp; | Blank line used to visually separate logical sections. |
| 1030 |     latest_state = {k: v.detach().cpu() for k, v in final_model.state_dict().items()} | Assigns computed value to variable latest_state for later use. |
| 1031 |     if best_state is None: | Starts conditional logic; this block executes only when the condition is true. |
| 1032 |         best_state = latest_state | Assigns computed value to variable best_state for later use. |
| 1033 |         if not best_metrics: | Starts conditional logic; this block executes only when the condition is true. |
| 1034 |             best_metrics = { | Assigns computed value to variable best_metrics for later use. |
| 1035 |                 "accuracy": 0.0, | Continuation line that contributes one element/argument to a larger structure. |
| 1036 |                 "precision": 0.0, | Continuation line that contributes one element/argument to a larger structure. |
| 1037 |                 "recall": 0.0, | Continuation line that contributes one element/argument to a larger structure. |
| 1038 |                 "f1": 0.0, | Continuation line that contributes one element/argument to a larger structure. |
| 1039 |                 "kappa": 0.0, | Continuation line that contributes one element/argument to a larger structure. |
| 1040 |                 "loss": 0.0, | Continuation line that contributes one element/argument to a larger structure. |
| 1041 |             } | Closes a list/dict/function-call block opened on previous lines. |
| 1042 | &nbsp; | Blank line used to visually separate logical sections. |
| 1043 |     state_to_save = latest_state if training_status == "interrupted" else best_state | Assigns computed value to variable state_to_save for later use. |
| 1044 | &nbsp; | Blank line used to visually separate logical sections. |
| 1045 |     config.model_out.parent.mkdir(parents=True, exist_ok=True) | Continuation line in an ongoing expression, literal, or control-flow block. |
| 1046 |     torch.save( | Continuation line in an ongoing expression, literal, or control-flow block. |
| 1047 |         { | Continuation line in an ongoing expression, literal, or control-flow block. |
| 1048 |             "model_state_dict": state_to_save, | Continuation line that contributes one element/argument to a larger structure. |
| 1049 |             "best_model_state_dict": best_state, | Continuation line that contributes one element/argument to a larger structure. |
| 1050 |             "vocabulary": vocabulary.token_to_id, | Continuation line that contributes one element/argument to a larger structure. |
| 1051 |             "model_config": { | Continuation line in an ongoing expression, literal, or control-flow block. |
| 1052 |                 "embedding_dim": config.embedding_dim, | Continuation line that contributes one element/argument to a larger structure. |
| 1053 |                 "hidden_size": chosen_hidden, | Continuation line that contributes one element/argument to a larger structure. |
| 1054 |                 "dropout": chosen_dropout, | Continuation line that contributes one element/argument to a larger structure. |
| 1055 |                 "learning_rate": chosen_lr, | Continuation line that contributes one element/argument to a larger structure. |
| 1056 |                 "crf_regularization": chosen_crf_reg, | Continuation line that contributes one element/argument to a larger structure. |
| 1057 |                 "max_seq_len": config.max_seq_len, | Continuation line that contributes one element/argument to a larger structure. |
| 1058 |                 "temporal_dim": train_dataset.temporal_dim, | Continuation line that contributes one element/argument to a larger structure. |
| 1059 |                 "categorical_dim": train_dataset.categorical_dim, | Continuation line that contributes one element/argument to a larger structure. |
| 1060 |                 "contextual_dim": train_dataset.contextual_dim, | Continuation line that contributes one element/argument to a larger structure. |
| 1061 |             }, | Closes a list/dict/function-call block opened on previous lines. |
| 1062 |             "best_metrics": best_metrics, | Continuation line that contributes one element/argument to a larger structure. |
| 1063 |             "label_column": label_column, | Continuation line that contributes one element/argument to a larger structure. |
| 1064 |             "label_mode": "explicit" if label_column else "weak_supervision", | Continuation line that contributes one element/argument to a larger structure. |
| 1065 |             "training_status": training_status, | Continuation line that contributes one element/argument to a larger structure. |
| 1066 |             "last_completed_epoch": last_completed_epoch, | Continuation line that contributes one element/argument to a larger structure. |
| 1067 |             "epochs_requested": config.epochs, | Continuation line that contributes one element/argument to a larger structure. |
| 1068 |         }, | Closes a list/dict/function-call block opened on previous lines. |
| 1069 |         config.model_out, | Continuation line that contributes one element/argument to a larger structure. |
| 1070 |     ) | Closes a list/dict/function-call block opened on previous lines. |
| 1071 | &nbsp; | Blank line used to visually separate logical sections. |
| 1072 |     if training_status == "interrupted": | Starts conditional logic; this block executes only when the condition is true. |
| 1073 |         print("Saved interrupted checkpoint to:", config.model_out) | Prints runtime information to the console for monitoring/debugging. |
| 1074 |     else: | Fallback branch that executes when no previous condition matched. |
| 1075 |         print("Saved best model to:", config.model_out) | Prints runtime information to the console for monitoring/debugging. |
| 1076 |     print("Best validation metrics:", best_metrics) | Prints runtime information to the console for monitoring/debugging. |
| 1077 | &nbsp; | Blank line used to visually separate logical sections. |
| 1078 | &nbsp; | Blank line used to visually separate logical sections. |
| 1079 | def default_model_output_path() -&gt; Path: | Declares function default_model_output_path and begins its parameter signature. |
| 1080 |     """Build a unique checkpoint path using current date and time.""" | Single-line docstring that documents behavior for the surrounding function/class/module. |
| 1081 |     timestamp = datetime.now().strftime("%d%m%Y_%H%M%S") | Assigns computed value to variable timestamp for later use. |
| 1082 |     return Path("models") / f"adlm_bilstm_crf_{timestamp}.pt" | Returns a value to the caller from the current function. |
| 1083 | &nbsp; | Blank line used to visually separate logical sections. |
| 1084 | &nbsp; | Blank line used to visually separate logical sections. |
| 1085 | def parse_args() -&gt; TrainConfig: | Declares function parse_args and begins its parameter signature. |
| 1086 |     parser = argparse.ArgumentParser( | Creates the command-line parser object and starts its configuration. |
| 1087 |         description="ADLM-style bug duplicate detection model (BiLSTM + CRF + optional Dragonfly tuning)." | Assigns computed value to variable description for later use. |
| 1088 |     ) | Closes a list/dict/function-call block opened on previous lines. |
| 1089 |     parser.add_argument("--data_path", type=Path, default=Path("dataset") / "corpus_features.csv") | Registers one command-line argument in argparse configuration. |
| 1090 |     parser.add_argument( | Registers one command-line argument in argparse configuration. |
| 1091 |         "--model_out", | Continuation line that contributes one element/argument to a larger structure. |
| 1092 |         type=Path, | Assigns computed value to variable type for later use. |
| 1093 |         default=None, | Assigns computed value to variable default for later use. |
| 1094 |         help="Optional checkpoint path. If omitted, a timestamped filename is generated automatically.", | Assigns computed value to variable help for later use. |
| 1095 |     ) | Closes a list/dict/function-call block opened on previous lines. |
| 1096 |     parser.add_argument("--label_column", type=str, default=None) | Registers one command-line argument in argparse configuration. |
| 1097 | &nbsp; | Blank line used to visually separate logical sections. |
| 1098 |     parser.add_argument("--max_vocab_size", type=int, default=15000) | Registers one command-line argument in argparse configuration. |
| 1099 |     parser.add_argument("--max_seq_len", type=int, default=128) | Registers one command-line argument in argparse configuration. |
| 1100 |     parser.add_argument("--embedding_dim", type=int, default=128) | Registers one command-line argument in argparse configuration. |
| 1101 |     parser.add_argument("--hidden_size", type=int, default=128) | Registers one command-line argument in argparse configuration. |
| 1102 |     parser.add_argument("--dropout", type=float, default=0.30) | Registers one command-line argument in argparse configuration. |
| 1103 |     parser.add_argument("--learning_rate", type=float, default=1e-3) | Registers one command-line argument in argparse configuration. |
| 1104 |     parser.add_argument("--crf_regularization", type=float, default=1e-4) | Registers one command-line argument in argparse configuration. |
| 1105 | &nbsp; | Blank line used to visually separate logical sections. |
| 1106 |     parser.add_argument("--batch_size", type=int, default=16) | Registers one command-line argument in argparse configuration. |
| 1107 |     parser.add_argument("--epochs", type=int, default=8) | Registers one command-line argument in argparse configuration. |
| 1108 |     parser.add_argument("--val_size", type=float, default=0.2) | Registers one command-line argument in argparse configuration. |
| 1109 |     parser.add_argument("--seed", type=int, default=42) | Registers one command-line argument in argparse configuration. |
| 1110 |     parser.add_argument("--use_dragonfly", action="store_true") | Registers one command-line argument in argparse configuration. |
| 1111 |     parser.add_argument("--dragonfly_population", type=int, default=5) | Registers one command-line argument in argparse configuration. |
| 1112 |     parser.add_argument("--dragonfly_iterations", type=int, default=2) | Registers one command-line argument in argparse configuration. |
| 1113 |     parser.add_argument("--dragonfly_inner_epochs", type=int, default=2) | Registers one command-line argument in argparse configuration. |
| 1114 | &nbsp; | Blank line used to visually separate logical sections. |
| 1115 |     args = parser.parse_args() | Parses CLI input into a structured args object. |
| 1116 |     resolved_model_out = args.model_out if args.model_out is not None else default_model_output_path() | Assigns computed value to variable resolved_model_out for later use. |
| 1117 |     return TrainConfig( | Returns a value to the caller from the current function. |
| 1118 |         data_path=args.data_path, | Assigns computed value to variable data_path for later use. |
| 1119 |         model_out=resolved_model_out, | Assigns computed value to variable model_out for later use. |
| 1120 |         label_column=args.label_column, | Assigns computed value to variable label_column for later use. |
| 1121 |         max_vocab_size=args.max_vocab_size, | Assigns computed value to variable max_vocab_size for later use. |
| 1122 |         max_seq_len=args.max_seq_len, | Assigns computed value to variable max_seq_len for later use. |
| 1123 |         embedding_dim=args.embedding_dim, | Assigns computed value to variable embedding_dim for later use. |
| 1124 |         hidden_size=args.hidden_size, | Assigns computed value to variable hidden_size for later use. |
| 1125 |         dropout=args.dropout, | Assigns computed value to variable dropout for later use. |
| 1126 |         learning_rate=args.learning_rate, | Assigns computed value to variable learning_rate for later use. |
| 1127 |         crf_regularization=args.crf_regularization, | Assigns computed value to variable crf_regularization for later use. |
| 1128 |         batch_size=args.batch_size, | Assigns computed value to variable batch_size for later use. |
| 1129 |         epochs=args.epochs, | Assigns computed value to variable epochs for later use. |
| 1130 |         val_size=args.val_size, | Assigns computed value to variable val_size for later use. |
| 1131 |         seed=args.seed, | Assigns computed value to variable seed for later use. |
| 1132 |         use_dragonfly=args.use_dragonfly, | Assigns computed value to variable use_dragonfly for later use. |
| 1133 |         dragonfly_population=args.dragonfly_population, | Assigns computed value to variable dragonfly_population for later use. |
| 1134 |         dragonfly_iterations=args.dragonfly_iterations, | Assigns computed value to variable dragonfly_iterations for later use. |
| 1135 |         dragonfly_inner_epochs=args.dragonfly_inner_epochs, | Assigns computed value to variable dragonfly_inner_epochs for later use. |
| 1136 |     ) | Closes a list/dict/function-call block opened on previous lines. |
| 1137 | &nbsp; | Blank line used to visually separate logical sections. |
| 1138 | &nbsp; | Blank line used to visually separate logical sections. |
| 1139 | if __name__ == "__main__": | Starts conditional logic; this block executes only when the condition is true. |
| 1140 |     train_config = parse_args() | Assigns computed value to variable train_config for later use. |
| 1141 |     run_training(train_config) | Function or method call that triggers computation or side effects. |
| 1142 | &nbsp; | Blank line used to visually separate logical sections. |
| 1143 | &nbsp; | Blank line used to visually separate logical sections. |
