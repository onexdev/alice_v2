#!/usr/bin/env python3
"""
ALICE - Advanced Legitimate Intelligence Crypto Explorer v2.0
Complete BSC Token Transfer Scanner System - Production Ready

Usage:
    python alice_scanner.py sc <wallet_address> p Vv <output_file>  # Full version
    python alice_scanner.py sc <wallet_address> p Vf <output_file>  # From only
    python alice_scanner.py h                                       # Show help

Author: Enterprise Development Team
License: MIT Professional
GitHub: https://github.com/enterprise-dev/alice-scanner
Version: 2.0.0 Final
"""

import sys
import os
import json
import time
import asyncio
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import configparser

# Check Python version
if sys.version_info < (3, 8):
    print("ERROR: Python 3.8 or higher is required")
    print(f"Current version: {sys.version}")
    sys.exit(1)

# Check and import required dependencies
try:
    import aiohttp
except ImportError:
    print("ERROR: aiohttp is required. Install with: pip install aiohttp")
    sys.exit(1)

try:
    import colorama
    from colorama import Fore, Style, init
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    print("WARNING: colorama not found. Install with: pip install colorama for colored output")
    COLORAMA_AVAILABLE = False

# ============================================================================
# COLOR DEFINITIONS AND UTILITIES
# ============================================================================
class Colors:
    if COLORAMA_AVAILABLE:
        SUCCESS = Fore.LIGHTGREEN_EX
        ERROR = Fore.LIGHTRED_EX
        WARNING = Fore.YELLOW
        INFO = Fore.BLUE
        STEP = Fore.CYAN
        HIGHLIGHT = Fore.MAGENTA
        NORMAL = Fore.WHITE
        BOLD = Style.BRIGHT
        RESET = Style.RESET_ALL
    else:
        SUCCESS = ERROR = WARNING = INFO = STEP = HIGHLIGHT = NORMAL = BOLD = RESET = ""

# ============================================================================
# EXCEPTION CLASSES
# ============================================================================
class ALICEException(Exception):
    def __init__(self, message: str, error_code: str = "ALICE_ERROR"):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
    def __str__(self):
        return f"[{self.error_code}] {self.message}"

class ValidationError(ALICEException):
    def __init__(self, message: str):
        super().__init__(message, "VALIDATION_ERROR")

class ConfigurationError(ALICEException):
    def __init__(self, message: str):
        super().__init__(message, "CONFIG_ERROR")

class APIError(ALICEException):
    def __init__(self, message: str, status_code: int = None):
        super().__init__(message, "API_ERROR")
        self.status_code = status_code

class ScanError(ALICEException):
    def __init__(self, message: str):
        super().__init__(message, "SCAN_ERROR")

# ============================================================================
# DATA STRUCTURES
# ============================================================================
@dataclass
class TransactionResult:
    transaction_hash: str
    method: str
    age: str
    from_address: str
    to_address: str
    token_info: str
    raw_data: Dict

# ============================================================================
# SECURITY AND UTILITIES
# ============================================================================
class SecurityManager:
    @staticmethod
    def validate_wallet_address(address: str) -> bool:
        return bool(re.match(r'^0x[a-fA-F0-9]{40}$', address or ""))
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        if not filename: return "output"
        sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', filename)
        sanitized = re.sub(r'\.{2,}', '.', sanitized).strip('. ')
        return sanitized[:100] or "output"
    @staticmethod
    def sanitize_input(input_str: str, max_length: int = 100) -> str:
        if not input_str: return ""
        s = re.sub(r'[<>"\';\\&|`$]', '', str(input_str))
        return re.sub(r'[\x00-\x1f\x7f]', '', s).strip()[:max_length]

class RateLimiter:
    def __init__(self, max_requests: int = 5, time_window: float = 1.0):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: List[float] = []
        self.lock = asyncio.Lock()
    async def acquire(self):
        async with self.lock:
            now = time.time()
            self.requests = [t for t in self.requests if now - t < self.time_window]
            if len(self.requests) >= self.max_requests:
                oldest = min(self.requests)
                wait = self.time_window - (now - oldest)
                if wait > 0:
                    await asyncio.sleep(wait)
                    now = time.time()
                    self.requests = [t for t in self.requests if now - t < self.time_window]
            self.requests.append(now)

# ============================================================================
# LOGGING SYSTEM
# ============================================================================
class Logger:
    def __init__(self, name: str = "ALICE"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        for h in list(self.logger.handlers): self.logger.removeHandler(h)
        fh = logging.FileHandler(self.log_dir / f"alice_{datetime.now():%Y%m%d}.log", encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s','%Y-%m-%d %H:%M:%S'))
        self.logger.addHandler(fh)
        self.log_step("INIT", "Logger initialized")
    def log_step(self, step: str, message: str):
        self.logger.info(f"[{step}] {message}")
        print(f"{Colors.STEP}[{step}]{Colors.NORMAL} {message}")
    def log_error(self, component: str, message: str):
        self.logger.error(f"[{component}] ERROR: {message}")
    def log_warning(self, component: str, message: str):
        self.logger.warning(f"[{component}] WARNING: {message}")

# ============================================================================
# INPUT VALIDATION
# ============================================================================
class InputValidator:
    def __init__(self):
        self.wallet_pattern = re.compile(r'^0x[a-fA-F0-9]{40}$')
        self.filename_pattern = re.compile(r'^[\w\.\-]+$')
    def validate_command_arguments(self, args: List[str]) -> Tuple[str, str, str, str]:
        if len(args) < 2: raise ValidationError("Insufficient arguments")
        if args[1].lower() == 'h': return "help", "", "", ""
        if len(args) < 6:
            raise ValidationError("Expected: python alice_scanner.py sc <wallet_address> p <version> <output_file>")
        cmd, wallet, p, version, outfile = args[1], args[2], args[3], args[4], args[5]
        if cmd.lower() != 'sc': raise ValidationError("Invalid command. Use 'sc'")
        if not SecurityManager.validate_wallet_address(wallet):
            raise ValidationError("Invalid wallet address")
        if p.lower() != 'p': raise ValidationError("Invalid parameter. Use 'p'")
        if version not in ['Vv', 'Vf']:
            raise ValidationError("Invalid version. Use 'Vv' or 'Vf'")
        fn = SecurityManager.sanitize_filename(outfile)
        if not fn.endswith('.txt'): fn += '.txt'
        return wallet.lower(), version, fn, "scan"

# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================
class ConfigManager:
    def __init__(self):
        self.config_file = Path('config.ini')
        self.credentials_dir = Path('credentials')
        self.credentials_file = self.credentials_dir / 'etherscan_key.json'
        self.api_key: str = ""
        self.config = None
    async def initialize(self):
        self._ensure_dirs()
        self._load_or_create_config()
        await self._load_credentials()
    def _ensure_dirs(self):
        for d in ['result','logs','credentials']:
            Path(d).mkdir(exist_ok=True)
    def _load_or_create_config(self):
        if not self.config_file.exists():
            cfg = configparser.ConfigParser()
            cfg['API'] = {
                'base_url': 'https://api.etherscan.io/v2/api',
                'rate_limit': '5',
                'timeout': '30',
                'max_retries': '3'
            }
            cfg['SCANNER'] = {
                'max_transactions': '1000',
                'concurrent_requests': '3'
            }
            with open(self.config_file,'w') as f: cfg.write(f)
        self.config = configparser.ConfigParser()
        self.config.read(self.config_file)
    async def _load_credentials(self):
        if not self.credentials_file.exists():
            tpl = {
                "api_key": "YOUR_ETHERSCAN_API_KEY_HERE",
                "instructions": [
                    "1. Visit https://etherscan.io/myapikey",
                    "2. Create or copy your API key",
                    "3. Replace YOUR_ETHERSCAN_API_KEY_HERE with your key"
                ]
            }
            with open(self.credentials_file,'w') as f: json.dump(tpl,f,indent=2)
            raise ConfigurationError("API key not configured. Edit credentials/etherscan_key.json")
        with open(self.credentials_file) as f:
            creds = json.load(f)
        key = creds.get('api_key','').strip()
        if not key or key == 'YOUR_ETHERSCAN_API_KEY_HERE':
            raise ConfigurationError("Invalid API key. Edit credentials/etherscan_key.json")
        self.api_key = key
    def get(self, section: str, option: str, default: str = "") -> str:
        return self.config.get(section, option, fallback=default)

# ============================================================================
# BSC SCANNER ENGINE
# ============================================================================
class BSCScanner:
    def __init__(self, cfg: ConfigManager, log: Logger):
        self.cfg = cfg
        self.logger = log
        self.base_url = self.cfg.get('API','base_url')
        self.rate_limiter = RateLimiter(int(self.cfg.get('API','rate_limit','5')))
        self.session = None
        self.initialized = False
    async def initialize(self):
        timeout = aiohttp.ClientTimeout(total=int(self.cfg.get('API','timeout','30')))
        self.session = aiohttp.ClientSession(timeout=timeout,headers={
            'User-Agent':'ALICE-Scanner/2.0.0','Accept':'application/json'
        })
        self.initialized = True
        self.logger.log_step("SCANNER","Initialized using V2 multi-chain API")
    async def cleanup(self):
        if self.session and not self.session.closed:
            await self.session.close()
    async def scan_token_transfers(self, wallet: str) -> List[TransactionResult]:
        if not self.initialized:
            raise ScanError("Scanner not initialized")
        await self.rate_limiter.acquire()
        params = {
            'module':'account',
            'action':'tokentx',
            'address':wallet,
            'page':'1','offset':'1000','sort':'desc',
            'chainId':'56',
            'apikey': self.cfg.api_key
        }
        async with self.session.get(self.base_url, params=params) as resp:
            data = await resp.json()
        return await self._process_response(data)
    async def _process_response(self, data: Dict) -> List[TransactionResult]:
        if data.get('status') != '1':
            msg = data.get('message','').lower()
            if 'invalid api key' in msg:
                raise APIError("Invalid API key")
            if 'rate limit' in msg:
                raise APIError("Rate limit exceeded")
            raise APIError(f"API Error: {data.get('message')}")
        txs = data.get('result',[])
        self.logger.log_step("PROCESS",f"Got {len(txs)} transactions")
        results = []
        for tx in txs:
            r = self._process_transaction(tx)
            if r: results.append(r)
        return results
    def _process_transaction(self, tx: Dict) -> Optional[TransactionResult]:
        h = tx.get('hash','')
        if len(h)!=66: return None
        sig = tx.get('input','')[:10]
        method = {'0xa9059cbb':'transfer','0x23b872dd':'transferFrom'}.get(sig,'other')
        try:
            age = datetime.utcfromtimestamp(int(tx.get('timeStamp','0'))).strftime('%Y-%m-%d %H:%M:%S UTC')
        except:
            age = 'Unknown'
        info = f"{tx.get('tokenName','Unknown')} ({tx.get('tokenSymbol','UNK')})"
        return TransactionResult(h, method, age, tx.get('from','').lower(), tx.get('to','').lower(), info, tx)
    def format_results(self, results: List[TransactionResult], version: str) -> List[str]:
        self.logger.log_step("FORMAT",f"Formatting {len(results)} records as {version}")
        if version=='Vv':
            return [f"{r.transaction_hash}|{r.method}|{r.age}|{r.from_address}|{r.to_address}|{r.token_info}" for r in results]
        return sorted({r.from_address for r in results})
    async def save_results(self, lines: List[str], fname: str) -> str:
        Path('result').mkdir(exist_ok=True)
        out = Path('result')/fname
        with open(out,'w',encoding='utf-8') as f:
            for l in lines: f.write(l+"\n")
        self.logger.log_step("SAVE",f"Saved to {out}")
        return str(out)

# ============================================================================
# TERMINAL INTERFACE
# ============================================================================
class TerminalInterface:
    def __init__(self, log: Logger):
        self.logger = log
    def show_banner(self):
        print(f"""{Colors.HIGHLIGHT}
┌───────────────────────── ALICE v2.0 ─────────────────────────┐
│ Advanced Legitimate Intelligence Crypto Explorer            │
└────────────────────────────────────────────────────────────┘{Colors.NORMAL}""")
    def get_user_confirmation(self) -> bool:
        try:
            ans = input(f"{Colors.WARNING}Proceed with scan? (y/n): {Colors.NORMAL}").lower().strip()
            return ans in ('y','yes','ya')
        except KeyboardInterrupt:
            print(f"\n{Colors.WARNING}Cancelled{Colors.NORMAL}")
            return False
    def show_help(self):
        print(f"""{Colors.INFO}
Usage: python alice_scanner.py sc <wallet_address> p <Vv|Vf> <output_file>
Get your API key at: https://etherscan.io/myapikey (multi-chain V2){Colors.NORMAL}""")
    def show_error(self, msg: str):
        print(f"{Colors.ERROR}[ERROR] {msg}{Colors.NORMAL}")
    def show_success(self, total: int, out_count: int, path: str, t: float):
        print(f"""{Colors.SUCCESS}
Scan complete:
  Transactions: {total}
  Records: {out_count}
  File: {path}
  Time: {t:.2f}s
{Colors.NORMAL}""")

# ============================================================================
# MAIN APPLICATION
# ============================================================================
class ALICEApplication:
    def __init__(self):
        self.logger = Logger()
        self.terminal = TerminalInterface(self.logger)
        self.validator = InputValidator()
        self.config = ConfigManager()
        self.scanner: BSCScanner = None  # type: ignore
    async def initialize(self):
        self.logger.log_step("INIT","Initializing system")
        await self.config.initialize()
        self.scanner = BSCScanner(self.config, self.logger)
        await self.scanner.initialize()
    def parse_args(self, args):
        return self.validator.validate_command_arguments(args)
    async def execute_scan(self, wallet, version, outfile):
        start = time.time()
        self.terminal.show_banner()
        if not self.terminal.get_user_confirmation():
            return False
        self.logger.log_step("SCAN",f"Wallet: {wallet}")
        results = await self.scanner.scan_token_transfers(wallet)
        if not results:
            self.terminal.show_error("No transfers found")
            return False
        lines = self.scanner.format_results(results, version)
        path = await self.scanner.save_results(lines, outfile)
        self.terminal.show_success(len(results), len(lines), path, time.time()-start)
        return True
    async def run(self, argv):
        try:
            await self.initialize()
            wallet, version, outfile, cmd = self.parse_args(argv)
            if cmd=="help":
                self.terminal.show_help()
                return True
            return await self.execute_scan(wallet, version, outfile)
        except ALICEException as e:
            self.terminal.show_error(str(e))
            return False

if __name__ == "__main__":
    success = asyncio.run(ALICEApplication().run(sys.argv))
    sys.exit(0 if success else 1)
