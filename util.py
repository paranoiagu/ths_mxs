import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import tushare as ts


DATE_FMT = "%Y%m%d"


def iter_date_strings(start_date: str, end_date: str) -> list[str]:
    start_dt = datetime.strptime(start_date, DATE_FMT)
    end_dt = datetime.strptime(end_date, DATE_FMT)
    if start_dt > end_dt:
        return []

    dates: list[str] = []
    current = start_dt
    while current <= end_dt:
        dates.append(current.strftime(DATE_FMT))
        current += timedelta(days=1)
    return dates


def shift_date_str(date_str: str, days: int) -> str:
    current = datetime.strptime(date_str, DATE_FMT)
    return (current + timedelta(days=days)).strftime(DATE_FMT)


def date_str_today() -> str:
    return datetime.now().strftime(DATE_FMT)


def normalize_date_str(date_str: str) -> str:
    return datetime.strptime(date_str, DATE_FMT).strftime(DATE_FMT)


def init_tushare():
    """Initialize tushare with the provided token."""
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        print("Error: TUSHARE_TOKEN not found")
        return None

    ts.set_token(token)
    pro = ts.pro_api()
    
    # Set timeout for tushare API requests (increase to 120 seconds)
    if hasattr(pro, 'api') and hasattr(pro.api, 'timeout'):
        pro.api.timeout = 120

    if pro is None:
        print("未能初始化Tushare，请检查TUSHARE_TOKEN环境变量")
    
    return pro


def api_call_with_retry(api_func, pro_api_instance, max_retries: int = 5, retry_delay: int = 10, timeout: int = 120, fields=None, **kwargs):
    """Call tushare API with retry mechanism and timeout handling.
    
    Args:
        api_func: The tushare API function to call
        pro_api_instance: The tushare pro_api instance (needed to set timeout)
        max_retries: Maximum number of retry attempts
        retry_delay: Delay in seconds between retries
        timeout: Request timeout in seconds
        **kwargs: Arguments to pass to the API function
        
    Returns:
        Result from the API call
        
    Raises:
        Exception: If all retries fail
    """
    import requests
    
    # Set timeout for the pro_api instance's underlying requests session
    if hasattr(pro_api_instance, 'api') and hasattr(pro_api_instance.api, 'timeout'):
        pro_api_instance.api.timeout = timeout
    
    for attempt in range(1, max_retries + 1):
        try:
            if fields is None:
                result = api_func(**kwargs, fields=["trade_date", "close", "open", "high", "low", "vol"])
            else:
                result = api_func(**kwargs, fields=fields)
            api_name = getattr(api_func, '__name__', None) or getattr(api_func, '__func__', None)
            if api_name and hasattr(api_name, '__name__'):
                api_name = api_name.__name__
            elif api_name:
                api_name = str(api_name)
            else:
                api_name = str(api_func)
            print(f"✅ API 调用成功 [{api_name}]")
            return result
            
        except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout, 
                requests.exceptions.ConnectionError) as e:
            if attempt < max_retries:
                wait_time = retry_delay * attempt
                print(f"⚠️ 网络超时错误 (尝试 {attempt}/{max_retries})，等待 {wait_time} 秒后重试...")
                print(f"错误详情: {str(e)}")
                time.sleep(wait_time)
            else:
                print(f"❌ 所有重试尝试均失败")
                raise
        except Exception as e:
            # Check if it's a timeout-related error in the error message
            error_str = str(e).lower()
            if 'timeout' in error_str or 'timed out' in error_str or 'read timeout' in error_str:
                if attempt < max_retries:
                    wait_time = retry_delay * attempt
                    print(f"⚠️ 网络超时错误 (尝试 {attempt}/{max_retries})，等待 {wait_time} 秒后重试...")
                    print(f"错误详情: {str(e)}")
                    time.sleep(wait_time)
                else:
                    print(f"❌ 所有重试尝试均失败")
                    raise
            else:
                # For other errors, also retry
                if attempt < max_retries:
                    wait_time = retry_delay * attempt
                    print(f"⚠️ API 调用错误 (尝试 {attempt}/{max_retries})，等待 {wait_time} 秒后重试...")
                    print(f"错误详情: {str(e)}")
                    time.sleep(wait_time)
                else:
                    print(f"❌ 所有重试尝试均失败")
                    raise
    
    raise Exception("所有重试尝试均失败")


def save_to_file(filename, content):
    try:
        # 确保目录存在
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        # 先删除已存在的文件（确保完全覆盖）
        if Path(filename).exists():
            Path(filename).unlink()
            
        # 写入新内容
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"数据已保存到 {filename}")
    except Exception as e:
        print(f"保存数据失败: {e}")


def get_last_range_dates() -> tuple[str, str]:
    """获取最近一个周期的开始和结束日期。（目前先取最近的300天）

    Returns:
        tuple[str, str]: (start_date, end_date) in 'YYYYMMDD' format
    """
    today = datetime.now()
    last_day = today - timedelta(days=300)

    start_date = last_day.strftime("%Y%m%d")
    end_date = today.strftime("%Y%m%d")

    return start_date, end_date


def read_file(file_path):
    try:
        file_path = Path(__file__).resolve().parent / file_path
        if not file_path.exists():
            print(f"文件不存在：{file_path}")
            return ""
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        print(f"读取文件失败：{e}")
        return ""
