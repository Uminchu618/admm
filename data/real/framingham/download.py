#!/usr/bin/env python3
"""
Kaggle Framingham Heart Study データセットをダウンロードするスクリプト

前提条件:
  - Kaggle API認証が設定されていること
  - .env に認証情報が存在すること
  - pip install kaggle python-dotenv
  
使用方法:
  python download.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv


def load_kaggle_credentials(env_path: Path) -> tuple[str, str]:
    """
    .env ファイルからKaggle認証情報を読み込む
    
    Args:
        env_path: .env ファイルのパス
        
    Returns:
        (username, api_key) のタプル
    """
    if not env_path.exists():
        print(f"❌ .env ファイルが見つかりません: {env_path}")
        print(f"   以下のコマンドで .env.example をコピーしてください:")
        print(f"   cp .env.example .env")
        sys.exit(1)
    
    load_dotenv(env_path)
    
    username = os.getenv("KAGGLE_USERNAME")
    api_key = os.getenv("KAGGLE_KEY")
    
    if not username or not api_key:
        print("❌ .env ファイルに KAGGLE_USERNAME または KAGGLE_KEY が設定されていません")
        print(f"   {env_path} を確認してください")
        sys.exit(1)
    
    return username, api_key


def download_framingham_dataset() -> None:
    """
    Framingham Heart Study データセットをダウンロード
    
    このスクリプトと同じディレクトリにダウンロードします
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("❌ Kaggleライブラリがインストールされていません")
        print("   実行: pip install kaggle python-dotenv")
        sys.exit(1)
    
    # スクリプトと同じディレクトリを出力先に
    script_dir = Path(__file__).parent
    env_path = script_dir / ".env"
    
    # .env から認証情報を読み込み
    username, api_key = load_kaggle_credentials(env_path)
    
    # 環境変数に設定
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = api_key
    
    # 認証
    try:
        api = KaggleApi()
        api.authenticate()
        print("✓ Kaggle認証成功")
    except Exception as e:
        print(f"❌ Kaggle認証失敗: {e}")
        print("   .env ファイルの KAGGLE_USERNAME と KAGGLE_KEY を確認してください")
        sys.exit(1)
    
    # ダウンロード先ディレクトリ作成
    script_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ 出力ディレクトリ: {script_dir.absolute()}")
    
    # ダウンロード
    print("⏳ ダウンロード中...")
    try:
        api.dataset_download_files(
            'aasheesh200/framingham-heart-study-dataset',
            path=str(script_dir),
            unzip=True
        )
        print("✓ ダウンロード完了")
        
        # ダウンロードしたファイルを表示
        files = [f for f in script_dir.glob("*") if f.is_file() and f.name not in [".env", ".env.example", ".gitignore", "download.py"]]
        print(f"\n📁 ダウンロード済みファイル ({len(files)}個):")
        for f in sorted(files):
            size = f.stat().st_size / (1024 * 1024)  # MB
            print(f"   - {f.name} ({size:.2f} MB)")
    
    except Exception as e:
        print(f"❌ ダウンロード失敗: {e}")
        sys.exit(1)


if __name__ == "__main__":
    download_framingham_dataset()
