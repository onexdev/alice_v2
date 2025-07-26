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
import aiohttp
try:
    import colorama
    from colorama import Fore, Style, init
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False

# ============================================================================
# COLOR DEFINITIONS AND UTILITIES
# ============================================================================
class Colors:
    if COLORAMA_AVAILABLE:
        SUCCESS = Fore.LIGHTGREEN_EX
        ERROR   = Fore.LIGHTRED_EX
        WARNING = Fore.YELLOW
        INFO    = Fore.BLUE
        STEP    = Fore.CYAN
        HIGHLIGHT = Fore.MAGENTA
        NORMAL  = Fore.WHITE
        BOLD    = Style.BRIGHT
        RESET   = Style.RESET_ALL
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

class ValidationError(ALICEException): pass
class ConfigurationError(ALICEException): pass
class APIError(ALICEException):
    def __init__(self, message: str, status_code: int = None):
        super().__init__(message, "API_ERROR")
        self.status_code = status_code
class ScanError(ALICEException): pass

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
        s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', filename)
        s = re.sub(r'\.{2,}', '.', s).strip('. ')
        return s[:100] or "output"

# ============================================================================
# RATE LIMITER
# ============================================================================
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
                wait = self.time_window - (now - min(self.requests))
                if wait > 0: await asyncio.sleep(wait)
                now = time.time()
                self.requests = [t for t in self.requests if now - t < self.time_window]
            self.requests.append(now)

# ============================================================================
# LOGGER
# ============================================================================
class Logger:
    def __init__(self, name: str = "ALICE"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.log_dir = Path("logs"); self.log_dir.mkdir(exist_ok=True)
        for h in list(self.logger.handlers): self.logger.removeHandler(h)
        fh = logging.FileHandler(self.log_dir/f"alice_{datetime.now():%Y%m%d}.log", encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s','%Y-%m-%d %H:%M:%S'))
        self.logger.addHandler(fh)
        self.log_step("INIT","Logger initialized")
    def log_step(self, step: str, msg: str):
        self.logger.info(f"[{step}] {msg}")
        print(f"{Colors.STEP}[{step}]{Colors.NORMAL} {msg}")
    def log_error(self, comp: str, msg: str):
        self.logger.error(f"[{comp}] ERROR: {msg}")
    def log_warning(self, comp: str, msg: str):
        self.logger.warning(f"[{comp}] WARNING: {msg}")

# ============================================================================
# INPUT VALIDATOR
# ============================================================================
class InputValidator:
    def __init__(self):
        self.wallet_rx = re.compile(r'^0x[a-fA-F0-9]{40}$')
    def validate(self, args: List[str]) -> Tuple[str,str,str,str]:
        if len(args)<2: raise ValidationError("Insufficient arguments")
        if args[1].lower()=="h": return "help","","",""
        if len(args)<6:
            raise ValidationError("Expected: python alice_scanner.py sc <wallet> p <Vv|Vf> <outfile>")
        cmd,wallet,p,ver,out = args[1],args[2],args[3],args[4],args[5]
        if cmd.lower()!="sc": raise ValidationError("Use 'sc' for scan")
        if not self.wallet_rx.match(wallet): raise ValidationError("Invalid wallet")
        if p.lower()!="p": raise ValidationError("Use 'p' for print")
        if ver not in ("Vv","Vf"): raise ValidationError("Version must be Vv or Vf")
        fn = SecurityManager.sanitize_filename(out)
        if not fn.endswith('.txt'): fn += '.txt'
        return wallet.lower(),ver,fn,"scan"

# ============================================================================
# CONFIG MANAGER (patch: default to V2 multi-chain)
# ============================================================================
class ConfigManager:
    def __init__(self):
        self.config_file = Path('config.ini')
        self.credentials_dir = Path('credentials')
        self.credentials_file = self.credentials_dir/'etherscan_key.json'
        self.api_key = ""
        self.config = None
    async def initialize(self):
        self._ensure_dirs()
        self._load_or_create_cfg()
        await self._load_creds()
    def _ensure_dirs(self):
        for d in ('result','logs','credentials'): Path(d).mkdir(exist_ok=True)
    def _load_or_create_cfg(self):
        if not self.config_file.exists():
            cfg = configparser.ConfigParser()
            cfg['API'] = {
                'base_url': 'https://api.etherscan.io/v2/api',
                'rate_limit': '5',
                'timeout': '30',
                'max_retries': '3'
            }
            with open(self.config_file,'w') as f: cfg.write(f)
        self.config = configparser.ConfigParser()
        self.config.read(self.config_file)
    async def _load_creds(self):
        if not self.credentials_file.exists():
            tpl = {
              "api_key":"YOUR_ETHERSCAN_API_KEY_HERE",
              "instructions":[
                "1. Visit https://etherscan.io/myapikey",
                "2. Copy your API key",
                "3. Replace YOUR_ETHERSCAN_API_KEY_HERE"
              ]
            }
            with open(self.credentials_file,'w') as f: json.dump(tpl,f,indent=2)
            raise ConfigurationError("Configure API key in credentials/etherscan_key.json")
        with open(self.credentials_file) as f:
            creds = json.load(f)
        key = creds.get('api_key','').strip()
        if not key or key=="YOUR_ETHERSCAN_API_KEY_HERE":
            raise ConfigurationError("Invalid API key in credentials/etherscan_key.json")
        self.api_key = key
    def get(self, section: str, option: str, default: str="") -> str:
        return self.config.get(section, option, fallback=default)

# ============================================================================
# SCANNER ENGINE (patch: add chainId=56)
# ============================================================================
class BSCScanner:
    def __init__(self, cfg: ConfigManager, log: Logger):
        self.cfg = cfg
        self.logger = log
        self.base_url = cfg.get('API','base_url')
        self.rate_limiter = RateLimiter(int(cfg.get('API','rate_limit','5')))
        self.session: Optional[aiohttp.ClientSession] = None
    async def initialize(self):
        timeout = aiohttp.ClientTimeout(total=int(self.cfg.get('API','timeout','30')))
        self.session = aiohttp.ClientSession(timeout=timeout, headers={
            'User-Agent':'ALICE-Scanner/2.0.0','Accept':'application/json'
        })
        self.logger.log_step("SCANNER","Initialized V2 multi-chain API")
    async def scan_token_transfers(self, wallet: str) -> List[TransactionResult]:
        await self.rate_limiter.acquire()
        params = {
            'module':'account','action':'tokentx','address':wallet,
            'page':'1','offset':'1000','sort':'desc',
            'chainId':'56',  # <<< patch: specify BSC
            'apikey': self.cfg.api_key
        }
        async with self.session.get(self.base_url, params=params) as resp:
            data = await resp.json()
        if data.get('status')!='1':
            msg = data.get('message','').lower()
            if 'invalid api key' in msg: raise APIError("Invalid API key")
            if 'rate limit'    in msg: raise APIError("Rate limit exceeded")
            raise APIError(f"API Error: {data.get('message')}")
        txs = data.get('result',[])
        self.logger.log_step("PROCESS",f"Got {len(txs)} TXs")
        results=[] 
        for tx in txs:
            h = tx.get('hash','')
            if len(h)!=66: continue
            sig = tx.get('input','')[:10]
            method = {'0xa9059cbb':'transfer','0x23b872dd':'transferFrom'}.get(sig,'other')
            try:
                age = datetime.utcfromtimestamp(int(tx.get('timeStamp','0'))).strftime('%Y-%m-%d %H:%M:%S UTC')
            except:
                age='Unknown'
            info = f"{tx.get('tokenName','Unknown')} ({tx.get('tokenSymbol','UNK')})"
            results.append(TransactionResult(h,method,age,tx.get('from',''),tx.get('to',''),info,tx))
        return results
    def format_results(self, results: List[TransactionResult], version: str) -> List[str]:
        self.logger.log_step("FORMAT",f"Formatting {len(results)} records as {version}")
        if version=='Vv':
            return [f"{r.transaction_hash}|{r.method}|{r.age}|{r.from_address}|{r.to_address}|{r.token_info}" for r in results]
        return sorted({r.from_address for r in results})
    async def save_results(self, lines: List[str], fname: str) -> str:
        out = Path('result')/fname
        with open(out,'w',encoding='utf-8') as f:
            for l in lines: f.write(l+'\n')
        self.logger.log_step("SAVE",f"Saved to {out}")
        return str(out)

# ============================================================================
# TERMINAL INTERFACE (unchanged)
# ============================================================================
class TerminalInterface:
    def __init__(self, logger: Logger):
        self.logger = logger
    def show_banner(self):
        banner = f"""{Colors.HIGHLIGHT}
┌─────────────────────────────────────────────────────────────┐
│                         A L I C E                           │
│            Advanced Legitimate Intelligence                 │
│              Crypto Explorer v2.0.0 Final                  │
└─────────────────────────────────────────────────────────────┘{Colors.NORMAL}"""
        print(banner)
    def get_user_confirmation(self):
        try:
            resp = input(f"{Colors.WARNING}Scan sekarang? (y/n): {Colors.NORMAL}").strip().lower()
            return resp in ('y','yes','ya')
        except KeyboardInterrupt:
            return False
    def show_help(self):
        print("Usage: python alice_scanner.py sc <wallet_address> p <version> <output_file>")
    def show_error(self, msg: str):
        print(f"{Colors.ERROR}[ERROR] {msg}{Colors.NORMAL}")
    def show_success(self,total,out_count,path,exec_time):
        print(f"{Colors.SUCCESS}Scan complete: TXs={total}, Records={out_count}, File={path}, Time={exec_time:.2f}s{Colors.NORMAL}")

# ============================================================================
# MAIN APP
# ============================================================================
class ALICEApplication:
    def __init__(self):
        self.logger = Logger()
        self.terminal = TerminalInterface(self.logger)
        self.validator = InputValidator()
        self.config = ConfigManager()
        self.scanner: Optional[BSCScanner] = None
    async def initialize(self):
        self.logger.log_step("INIT","Initializing...")
        await self.config.initialize()
        self.scanner = BSCScanner(self.config, self.logger)
        await self.scanner.initialize()
    def parse_args(self,args):
        return self.validator.validate(args)
    async def execute_scan(self,wallet,ver,outfile):
        start=time.time()
        self.terminal.show_banner()
        if not self.terminal.get_user_confirmation(): return False
        results = await self.scanner.scan_token_transfers(wallet)
        if not results:
            self.terminal.show_error("No token transfers found")
            return False
        lines = self.scanner.format_results(results, ver)
        path = await self.scanner.save_results(lines, outfile)
        self.terminal.show_success(len(results), len(lines), path, time.time()-start)
        return True
    async def run(self,argv):
        try:
            await self.initialize()
            wallet,ver,outfile,cmd = self.parse_args(argv)
            if cmd=="help":
                self.terminal.show_help(); return True
            return await self.execute_scan(wallet, ver, outfile)
        except ALICEException as e:
            self.terminal.show_error(str(e))
            return False

if __name__=="__main__":
    success = asyncio.run(ALICEApplication().run(sys.argv))
    sys.exit(0 if success else 1)
