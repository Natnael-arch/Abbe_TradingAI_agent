#!/usr/bin/env python3
"""
Script to restore sensitive configuration values from backup file.
This should be run AFTER cloning the repository to restore your API keys.
IMPORTANT: DO NOT run this script before pushing to GitHub!
"""

import yaml
import os
from pathlib import Path


def restore_config():
    """Restore sensitive configuration from backup file."""
    backup_file = Path("config/config_backup.yaml")
    config_file = Path("config/config.yaml")
    
    if not backup_file.exists():
        print("❌ Backup file not found: config/config_backup.yaml")
        print("Please ensure you have the backup file with your original API keys.")
        return False
    
    if not config_file.exists():
        print("❌ Config file not found: config/config.yaml")
        print("Please copy config.example.yaml to config.yaml first.")
        return False
    
    try:
        # Load backup file
        with open(backup_file, 'r') as f:
            backup_data = yaml.safe_load(f)
        
        # Load current config
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Restore exchange API keys
        if 'original_config' in backup_data and 'exchange' in backup_data['original_config']:
            config_data['exchange']['api_key'] = backup_data['original_config']['exchange']['api_key']
            config_data['exchange']['secret'] = backup_data['original_config']['exchange']['secret']
            print("✅ Restored exchange API keys")
        
        # Restore deployment Gaia API key
        if 'original_config' in backup_data and 'deployment' in backup_data['original_config']:
            config_data['deployment']['gaia_api_key'] = backup_data['original_config']['deployment']['gaia_api_key']
            print("✅ Restored deployment Gaia API key")
        
        # Restore recall API key
        if 'original_config' in backup_data and 'recall' in backup_data['original_config']:
            config_data['recall']['api_key'] = backup_data['original_config']['recall']['api_key']
            print("✅ Restored recall API key")
        
        # Restore Gaia AI API key
        if 'original_config' in backup_data and 'gaia' in backup_data['original_config']:
            config_data['gaia']['api_key'] = backup_data['original_config']['gaia']['api_key']
            print("✅ Restored Gaia AI API key")
        
        # Save updated config
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        
        print("✅ Configuration restored successfully!")
        print("⚠️  Remember: Never commit the backup file to GitHub!")
        return True
        
    except Exception as e:
        print(f"❌ Error restoring configuration: {e}")
        return False


def restore_coinbase_config():
    """Restore Coinbase configuration from backup file."""
    backup_file = Path("config/config_backup.yaml")
    coinbase_config_file = Path("config/config_coinbase.yaml")
    
    if not backup_file.exists():
        print("❌ Backup file not found: config/config_backup.yaml")
        return False
    
    if not coinbase_config_file.exists():
        print("❌ Coinbase config file not found: config/config_coinbase.yaml")
        return False
    
    try:
        # Load backup file
        with open(backup_file, 'r') as f:
            backup_data = yaml.safe_load(f)
        
        # Load current coinbase config
        with open(coinbase_config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Restore Gaia API key for Coinbase config
        if 'original_coinbase_config' in backup_data and 'gaia' in backup_data['original_coinbase_config']:
            config_data['gaia']['api_key'] = backup_data['original_coinbase_config']['gaia']['api_key']
            print("✅ Restored Coinbase Gaia API key")
        
        # Save updated config
        with open(coinbase_config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        
        print("✅ Coinbase configuration restored successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error restoring Coinbase configuration: {e}")
        return False


if __name__ == "__main__":
    print("🔧 Restoring sensitive configuration from backup...")
    print("⚠️  WARNING: This script restores real API keys. Only run after cloning from GitHub!")
    print("   DO NOT run this script before pushing to GitHub!\n")
    
    success1 = restore_config()
    success2 = restore_coinbase_config()
    
    if success1 and success2:
        print("\n🎉 All configurations restored successfully!")
        print("You can now run your trading agent with your original API keys.")
        print("⚠️  Remember: Never commit the backup file to GitHub!")
    else:
        print("\n⚠️  Some configurations could not be restored.")
        print("Please check that you have the backup file with your original API keys.") 