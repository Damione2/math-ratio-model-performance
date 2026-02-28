#!/usr/bin/env python3
# guardian_cleanup.py
"""
guardian_cleanup.py - Safe cleanup script for Guardian training data and artifacts.

Run this BEFORE starting a fresh pipeline to ensure clean state.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# ANSI colors for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def get_size(path):
    """Calculate total size of file or directory in GB"""
    if path.is_file():
        return path.stat().st_size / (1024**3)
    elif path.is_dir():
        total = 0
        for item in path.rglob('*'):
            if item.is_file():
                total += item.stat().st_size
        return total / (1024**3)
    return 0

def list_contents(directory, indent="  "):
    """List all files and folders with sizes"""
    if not directory.exists():
        print(f"{indent}{Colors.YELLOW}(directory does not exist){Colors.RESET}")
        return 0, 0
    
    files = []
    dirs = []
    
    for item in sorted(directory.iterdir(), key=lambda x: x.name.lower()):
        if item.is_file():
            size_mb = item.stat().st_size / (1024**2)
            files.append((item.name, size_mb))
        elif item.is_dir():
            size_gb = get_size(item)
            dirs.append((item.name, size_gb))
    
    # Print directories first
    for name, size_gb in dirs:
        print(f"{indent}📁 {name}/  ({size_gb:.2f} GB)")
    
    # Then files
    for name, size_mb in files:
        print(f"{indent}📄 {name}  ({size_mb:.2f} MB)")
    
    return len(dirs), len(files)

def safe_delete(path, dry_run=True):
    """Safely delete a file or directory"""
    if not path.exists():
        return False
    
    size = get_size(path)
    
    if dry_run:
        action = "Would delete"
        color = Colors.YELLOW
    else:
        try:
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
            action = "Deleted"
            color = Colors.RED
        except Exception as e:
            print(f"    {Colors.RED}Error deleting {path}: {e}{Colors.RESET}")
            return False
    
    item_type = "📁" if path.is_dir() else "📄"
    print(f"    {color}{action}{Colors.RESET} {item_type} {path.name} ({size:.2f} GB)")
    return True

def cleanup_guardian(dry_run=True, preserve_merged=False, backup_models=False, force=False):
    """
    Main cleanup function.
    
    Args:
        dry_run: If True, only show what would be deleted
        preserve_merged: If True, keep 01_raw_data_merged.pkl
        backup_models: If True, move models to backup folder instead of deleting
        force: If True, skip confirmation prompts
    """
    
    # Paths from config.py
    artifacts_dir = Path("C:/guardian_artifacts")
    data_dir = Path("C:/guardian_data")
    
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}🧹 GUARDIAN CLEANUP SCRIPT{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
    
    mode = "DRY RUN (no changes)" if dry_run else "LIVE DELETE"
    print(f"\nMode: {Colors.CYAN}{mode}{Colors.RESET}")
    print(f"Preserve merged data: {Colors.GREEN if preserve_merged else Colors.RED}{preserve_merged}{Colors.RESET}")
    print(f"Backup models: {Colors.GREEN if backup_models else Colors.RED}{backup_models}{Colors.RESET}")
    
    # Check what exists
    artifacts_exists = artifacts_dir.exists()
    data_exists = data_dir.exists()
    
    total_size = 0
    if artifacts_exists:
        total_size += get_size(artifacts_dir)
    if data_exists:
        total_size += get_size(data_dir)
    
    print(f"\n{Colors.BOLD}Current disk usage:{Colors.RESET}")
    print(f"  C:/guardian_artifacts: {'EXISTS' if artifacts_exists else 'NOT FOUND'}")
    if artifacts_exists:
        print(f"    Size: {get_size(artifacts_dir):.2f} GB")
    print(f"  C:/guardian_data: {'EXISTS' if data_exists else 'NOT FOUND'}")
    if data_exists:
        print(f"    Size: {get_size(data_dir):.2f} GB")
    print(f"  {Colors.BOLD}Total: {total_size:.2f} GB{Colors.RESET}")
    
    # Show contents
    if artifacts_exists:
        print(f"\n{Colors.CYAN}Contents of C:/guardian_artifacts:{Colors.RESET}")
        list_contents(artifacts_dir)
    
    if data_exists:
        print(f"\n{Colors.CYAN}Contents of C:/guardian_data:{Colors.RESET}")
        list_contents(data_dir)
    
    # Confirmation (unless force or dry_run)
    if not dry_run and not force:
        print(f"\n{Colors.RED}{'='*70}{Colors.RESET}")
        print(f"{Colors.RED}⚠️  WARNING: This will PERMANENTLY DELETE all files above!{Colors.RESET}")
        print(f"{Colors.RED}{'='*70}{Colors.RESET}")
        response = input(f"\nType 'DELETE' to confirm: ")
        if response.strip() != "DELETE":
            print(f"\n{Colors.YELLOW}Cleanup cancelled.{Colors.RESET}")
            return False
    
    # Perform cleanup
    deleted_count = 0
    preserved_count = 0
    
    # Clean artifacts
    if artifacts_exists:
        print(f"\n{Colors.BOLD}Cleaning C:/guardian_artifacts...{Colors.RESET}")
        
        for item in artifacts_dir.iterdir():
            # Preserve merged data if requested
            if preserve_merged and item.name == "01_raw_data_merged.pkl":
                print(f"    {Colors.GREEN}Preserving{Colors.RESET} 📄 {item.name}")
                preserved_count += 1
                continue
            
            # Backup models if requested
            if backup_models and item.suffix == ".pth":
                backup_dir = artifacts_dir / "_backup_models"
                backup_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"{item.stem}_{timestamp}{item.suffix}"
                backup_path = backup_dir / backup_name
                
                if not dry_run:
                    shutil.move(str(item), str(backup_path))
                    print(f"    {Colors.YELLOW}Backed up{Colors.RESET} 📄 {item.name} → _backup_models/{backup_name}")
                else:
                    print(f"    Would backup 📄 {item.name} → _backup_models/{backup_name}")
                continue
            
            # Delete everything else
            if safe_delete(item, dry_run=dry_run):
                deleted_count += 1
    
    # Clean data
    if data_exists:
        print(f"\n{Colors.BOLD}Cleaning C:/guardian_data...{Colors.RESET}")
        
        for item in data_dir.iterdir():
            if safe_delete(item, dry_run=dry_run):
                deleted_count += 1
    
    # Summary
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}CLEANUP SUMMARY{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
    
    if dry_run:
        print(f"  Mode: {Colors.YELLOW}DRY RUN{Colors.RESET} (no files were actually deleted)")
        print(f"  Run with --execute to perform actual deletion")
    else:
        print(f"  Items deleted: {Colors.RED}{deleted_count}{Colors.RESET}")
        print(f"  Items preserved: {Colors.GREEN}{preserved_count}{Colors.RESET}")
        if backup_models:
            print(f"  Models backed up to: C:/guardian_artifacts/_backup_models/")
    
    # Show remaining space
    if not dry_run and (artifacts_exists or data_exists):
        remaining_size = 0
        if artifacts_dir.exists():
            remaining_size += get_size(artifacts_dir)
        if data_dir.exists():
            remaining_size += get_size(data_dir)
        print(f"  Remaining data: {remaining_size:.2f} GB")
    
    print(f"\n{Colors.GREEN}✅ Cleanup complete!{Colors.RESET}\n")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean Guardian training data and artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run - see what would be deleted
  python guardian_cleanup.py
  
  # Actually delete everything
  python guardian_cleanup.py --execute
  
  # Delete but preserve merged dataset
  python guardian_cleanup.py --execute --preserve-merged
  
  # Delete but backup model files
  python guardian_cleanup.py --execute --backup-models
  
  # Full clean with no prompts
  python guardian_cleanup.py --execute --force
        """
    )
    
    parser.add_argument(
        "--execute", 
        action="store_true",
        help="Actually perform deletion (default is dry-run)"
    )
    parser.add_argument(
        "--preserve-merged",
        action="store_true", 
        help="Keep 01_raw_data_merged.pkl if it exists"
    )
    parser.add_argument(
        "--backup-models",
        action="store_true",
        help="Move .pth model files to _backup_models/ instead of deleting"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt (use with caution!)"
    )
    
    args = parser.parse_args()
    
    # Default to dry-run unless --execute is specified
    dry_run = not args.execute
    
    success = cleanup_guardian(
        dry_run=dry_run,
        preserve_merged=args.preserve_merged,
        backup_models=args.backup_models,
        force=args.force
    )
    
    sys.exit(0 if success else 1)