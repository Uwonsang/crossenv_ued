"""
RunPod Multi-Process Deployment Script

Features:
- Multi-pod parallel execution with real-time dashboard
- Emergency cleanup mechanism for interrupted execution
- Signal handlers for SIGINT (Ctrl+C) and SIGTERM
- Automatic pod termination on script exit
- Activity logging with status tracking

Safety mechanisms:
1. All created pods are registered in a global tracker
2. Signal handlers catch Ctrl+C and kill signals
3. atexit ensures cleanup on normal termination
4. Each pod process unregisters on successful cleanup
5. Emergency cleanup terminates all tracked pods on failure
"""

import os
import time
import yaml
import argparse
import signal
import atexit
from enum import Enum
from pathlib import Path
from itertools import islice
from multiprocessing import Process, Manager
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout

import runpod
import requests
from dotenv import load_dotenv

base_url = 'https://rest.runpod.io/v1'

# Global tracking for cleanup
_active_pods = {}  # {pod_id: API_KEY}
_cleanup_lock = None
_console = Console()

defaults = {
    'name': f'pcg-{" ".join(time.ctime().split()[1:-1])}',
    'template_id': 'zkat0jqlju',
    'gpu': 'NVIDIA RTX A4000',
    'gpu_count': 1,
    'spot': False,
    'commands': []
}

# ---
command_folder_path = os.path.join(__file__.rstrip('train_in_runpod.py'), 'model')
config_folder_path = os.path.join(__file__.rstrip('train_in_runpod.py'), 'config')


# ---

class CmdType(Enum):
    TRAIN = "TRAIN"
    STOP = "STOP"
    WANDB_LOGIN = "WANDB_LOGIN"


def parse_args():
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument("-c", "--config", type=str, help="YAML config file path", default=f'config.yaml')
    conf_parser.add_argument("--configs", type=str, nargs='+', help="Multiple YAML config files for multi-pod execution")
    temp_args, remaining_argv = conf_parser.parse_known_args()

    config_path = os.path.join(config_folder_path, temp_args.config)

    if temp_args.config and Path(config_path).exists():
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

            pod_cfg = data.get('pod', {})
            opt_cfg = data.get('options', {})

            yaml_updates = {
                'template_id': pod_cfg.get('template_id'),
                'gpu': pod_cfg.get('gpu'),
                'gpu_count': pod_cfg.get('gpu_count'),
            }

            yaml_updates = {k: v for k, v in yaml_updates.items() if v is not None}
            commands = []
            for entry in data.get('runtime', {}).get('cmds', []):
                cmd_type = next(iter(entry))
                if cmd_type == CmdType.TRAIN.value:
                    commands.append((CmdType.TRAIN, dict(islice(entry.items(), 1, None))))
                else:
                    commands.append((CmdType(cmd_type), None))

            yaml_updates['commands'] = commands
            defaults.update(yaml_updates)

    parser = argparse.ArgumentParser(parents=[conf_parser])

    parser.add_argument('--name', type=str)

    parser.add_argument('--template-id', type=str)
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--gpu-count', type=int)
    parser.add_argument('--spot', action=argparse.BooleanOptionalAction)

    parser.add_argument('--timeout', action=argparse.BooleanOptionalAction)
    parser.add_argument('--time-limit', type=float)
    parser.add_argument('--terminate', action=argparse.BooleanOptionalAction)

    parser.set_defaults(**defaults)

    args = parser.parse_args()

    # If configs is specified, use it
    if temp_args.configs:
        args.configs = temp_args.configs

    return args


def create_pod_config_from_yaml(config_data, args):
    """Create pod configuration from yaml data"""
    pod_cfg = config_data.get('pod', {})

    cmd = []
    for entry in config_data.get('runtime', {}).get('cmds', []):
        cmd.append(entry)

    cmd_str = " && ".join(cmd) if cmd else ""

    # RunPod API configuration
    runpod_config = {
        'name': config_data.get('name', args.name),
        'templateId': pod_cfg.get('template_id', args.template_id),
        'gpuTypeIds': [pod_cfg.get('gpu', args.gpu)],
        'gpuCount': pod_cfg.get('gpu_count', args.gpu_count),
    }

    if pod_cfg.get('spot', args.spot):
        runpod_config['interruptible'] = True

    if cmd_str:
        runpod_config['dockerStartCmd'] = ['bash', '-lc', cmd_str]

    # Internal configuration for monitoring and termination
    internal_config = {
        'timeout': config_data.get('options', {}).get('timeout', getattr(args, 'timeout', False)),
        'time_limit': config_data.get('options', {}).get('time_limit', getattr(args, 'time_limit', 7200)),
        'terminate': config_data.get('options', {}).get('terminate', getattr(args, 'terminate', True)),
    }

    # Combine configurations
    combined_config = {**runpod_config, **internal_config}

    return combined_config


def cleanup_all_pods(signum=None, frame=None):
    """Emergency cleanup function to terminate all active pods"""
    global _active_pods, _cleanup_lock

    if not _active_pods:
        return

    _console.print("\n[bold red]🚨 Emergency cleanup initiated...[/bold red]")
    _console.print(f"[yellow]Terminating {len(_active_pods)} active pod(s)...[/yellow]")

    cleanup_results = []
    for pod_id, api_key in list(_active_pods.items()):
        try:
            runpod.api_key = api_key
            pod_info = runpod.get_pod(pod_id=pod_id)
            if pod_info:
                runpod.terminate_pod(pod_id=pod_id)
                cleanup_results.append((pod_id, 'success'))
                _console.print(f"[green]✓[/green] Terminated pod: {pod_id[:12]}...")
            else:
                cleanup_results.append((pod_id, 'already_gone'))
                _console.print(f"[dim]○[/dim] Pod already gone: {pod_id[:12]}...")
        except Exception as e:
            cleanup_results.append((pod_id, f'error: {str(e)[:30]}'))
            _console.print(f"[red]✗[/red] Failed to terminate {pod_id[:12]}...: {str(e)[:50]}")

    _active_pods.clear()

    success_count = sum(1 for _, status in cleanup_results if status == 'success')
    _console.print(f"\n[cyan]Cleanup complete: {success_count}/{len(cleanup_results)} pods terminated[/cyan]")

    # If called by signal, exit
    if signum is not None:
        _console.print("[yellow]Exiting...[/yellow]")
        os._exit(0)


def register_pod(pod_id, api_key):
    """Register a pod for cleanup tracking"""
    global _active_pods
    _active_pods[pod_id] = api_key


def unregister_pod(pod_id):
    """Unregister a pod from cleanup tracking"""
    global _active_pods
    _active_pods.pop(pod_id, None)


def run_single_pod(pod_config, status_dict, pod_id_key, API_KEY, log_queue):
    """Run a single pod and update its status"""
    runpod.api_key = API_KEY
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    base_url = 'https://rest.runpod.io/v1'
    
    pod_id = ''
    spinners = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    spinner_idx = 0

    def update_status(key, **kwargs):
        """Helper function to update status dict with proper syncing"""
        current = dict(status_dict.get(key, {}))

        # Log status changes
        old_status = current.get('status', '')
        new_status = kwargs.get('status', old_status)

        current.update(kwargs)
        status_dict[key] = current

        # Add log entry if status changed (excluding spinner updates)
        if 'status' in kwargs and old_status and new_status:
            # Remove spinner characters for comparison
            old_clean = old_status.split(' ', 1)[-1] if ' ' in old_status else old_status
            new_clean = new_status.split(' ', 1)[-1] if ' ' in new_status else new_status

            if old_clean != new_clean:
                timestamp = datetime.now().strftime('%H:%M:%S')
                pod_name = current.get('name', f'Pod{key}')
                log_msg = f"[{timestamp}] {pod_name}: {old_clean} → {new_clean}"
                if 'progress' in kwargs:
                    log_msg += f" ({kwargs['progress']})"
                log_queue.append(log_msg)

    try:
        # make name pcg-shortmd5-<timestamp> first
        hash_str = os.urandom(4).hex()
        date_str = datetime.now().strftime('%Y%m%d-%H%M%S')
        pod_name = f"pcg-{hash_str}-{date_str}"

        # Update status - initialize immediately with the pod name
        update_status(pod_id_key,
            status=f'{spinners[spinner_idx % len(spinners)]} CREATING',
            pod_id='',
            name=pod_name,
            start_time=datetime.now().strftime('%H:%M:%S'),
            runtime='0s',
            error='',
            progress='Requesting pod...'
        )
        spinner_idx += 1

        update_status(pod_id_key, progress='Sending request to RunPod API...')

        # RunPod API 요청용 설정 (내부 설정 제외)
        runpod_request = {
            'name': pod_name,
            'templateId': 'zkat0jqlju',
            'gpuTypeIds': ['NVIDIA RTX A4000'],
            'gpuCount': 1,
            'networkVolumeId': 'c4fddrd4z9'
        }

        # dockerStartCmd가 있으면 추가
        if 'dockerStartCmd' in pod_config:
            runpod_request['dockerStartCmd'] = pod_config['dockerStartCmd']

        # and stop pod
        if 'dockerStartCmd' in runpod_request:
            runpod_request['dockerStartCmd'][-1] += ' && runpodctl stop pod $RUNPOD_POD_ID'

        response = requests.post(f'{base_url}/pods', headers=headers, json=runpod_request)

        if response.status_code == 500:
            error_msg = response.json().get('error', 'Unknown error')
            update_status(pod_id_key,
                status='❌ ERROR',
                error=error_msg,
                progress='Failed'
            )
            return
        response.raise_for_status()
        
        pod_id = response.json().get('id', '')
        if not pod_id:
            update_status(pod_id_key,
                status='❌ ERROR',
                error='No pod id returned',
                progress='Failed'
            )
            return

        # Register pod for emergency cleanup
        register_pod(pod_id, API_KEY)

        update_status(pod_id_key,
            pod_id=pod_id,
            status=f'{spinners[spinner_idx % len(spinners)]} INITIALIZING',
            progress=f'Pod created: {pod_id[:8]}...'
        )
        spinner_idx += 1

        start_time = time.time()
        timeout = pod_config.get('timeout', False)
        time_limit = pod_config.get('time_limit', 3600)
        terminate = pod_config.get('terminate', True)
        
        # Monitor pod with spinner (Pod 생성은 보통 5분 정도 소요)
        # 스피너는 빠르게(0.25s), API 폴링은 느리게(3s)
        poll_interval = 3.0
        spinner_interval = 0.25
        last_poll = 0.0
        last_spinner = 0.0
        last_pod_info = None

        while True:
            now = time.time()
            elapsed = int(now - start_time)
            elapsed_min = elapsed // 60
            elapsed_sec = elapsed % 60
            runtime_str = f'{elapsed_min}m{elapsed_sec}s' if elapsed_min > 0 else f'{elapsed}s'

            # API 폴링은 주기적으로만 호출
            if now - last_poll >= poll_interval:
                last_poll = now
                pod_info = runpod.get_pod(pod_id=pod_id)
                if not pod_info:
                    update_status(pod_id_key,
                        status='💀 TERMINATED',
                        progress='Pod not found',
                        runtime=runtime_str
                    )
                    break
                last_pod_info = pod_info

                current_status = pod_info.get('desiredStatus', 'UNKNOWN')

                # Update progress based on actual pod status with time indicators
                if current_status == 'RUNNING':
                    status_label = 'RUNNING'
                    progress_msg = 'Training in progress...'
                elif current_status == 'EXITED':
                    update_status(pod_id_key,
                        status='✅ EXITED',
                        progress='Completed',
                        runtime=runtime_str
                    )
                    break
                elif current_status == 'CREATED':
                    status_label = 'STARTING'
                    # Pod 생성은 보통 5분 걸림
                    if elapsed < 60:
                        progress_msg = 'Starting... (usually takes ~5min)'
                    elif elapsed < 180:
                        progress_msg = f'Starting... ({elapsed_min}m elapsed)'
                    elif elapsed < 300:
                        progress_msg = f'Still starting... ({elapsed_min}m, normal)'
                    else:
                        progress_msg = f'Starting... ({elapsed_min}m, taking longer)'
                else:
                    status_label = current_status
                    progress_msg = f'Status: {current_status}'

            else:
                # 폴링하지 않는 사이에는 마지막 알려진 상태 라벨을 유지
                if last_pod_info:
                    current_status = last_pod_info.get('desiredStatus', 'UNKNOWN')
                    if current_status == 'RUNNING':
                        status_label = 'RUNNING'
                    elif current_status == 'CREATED':
                        status_label = 'STARTING'
                    else:
                        status_label = current_status
                else:
                    status_label = 'CREATING'
                progress_msg = None  # Don't update progress between polls

            # 스피너는 별도로 빠르게 업데이트
            if now - last_spinner >= spinner_interval:
                last_spinner = now
                spinner_char = spinners[spinner_idx % len(spinners)]
                spinner_idx += 1

                # RUNNING이면 🚀 사용, 그 외에는 스피너 사용
                if last_pod_info and last_pod_info.get('desiredStatus') == 'RUNNING':
                    status_str = '🚀 RUNNING'
                else:
                    status_str = f'{spinner_char} {status_label}'

                # 업데이트할 정보 준비
                update_dict = {
                    'status': status_str,
                    'runtime': runtime_str
                }
                if progress_msg is not None:
                    update_dict['progress'] = progress_msg

                update_status(pod_id_key, **update_dict)

            # 타임아웃 체크
            if timeout and elapsed > time_limit:
                update_status(pod_id_key,
                    status='⏱️ TIMEOUT',
                    progress=f'Exceeded {time_limit}s limit',
                    runtime=runtime_str
                )
                break
            
            time.sleep(0.1)  # 짧게 쉬어가며 스피너를 부드럽게 업데이트

        # Cleanup
        if terminate and pod_id:
            update_status(pod_id_key, progress='Terminating...')
            pod_info = runpod.get_pod(pod_id=pod_id)
            if pod_info:
                runpod.terminate_pod(pod_id=pod_id)
                time.sleep(1)
                update_status(pod_id_key,
                    status='🛑 TERMINATED',
                    progress='Cleaned up'
                )
            # Unregister from cleanup tracking
            unregister_pod(pod_id)

    except Exception as e:
        update_status(pod_id_key,
            status='❌ ERROR',
            error=str(e)[:100],
            progress='Failed with exception'
        )
        # Ensure pod is unregistered even on error
        if pod_id:
            unregister_pod(pod_id)


def generate_dashboard(status_dict, log_queue, max_logs=15):
    """Generate rich dashboard with status table and log panel"""
    # Create status table
    table = Table(title="RunPod Multi-Process Dashboard", show_header=True, header_style="bold magenta")
    table.add_column("Index", style="cyan", width=6)
    table.add_column("Name", style="green", width=20)
    table.add_column("Pod ID", style="yellow", width=15)
    table.add_column("Status", style="white", width=15)
    table.add_column("Progress", style="magenta", width=25)
    table.add_column("Runtime", style="blue", width=8)
    table.add_column("Start", style="white", width=8)
    table.add_column("Error", style="red", width=20)

    for idx in sorted(status_dict.keys()):
        info = status_dict[idx]
        status = info['status']
        
        # Status is already emoji-decorated from run_single_pod
        status_colored = status

        table.add_row(
            str(idx),
            info['name'][:19],
            info['pod_id'][:14] if info['pod_id'] else '-',
            status_colored,
            info.get('progress', '-')[:24],
            info['runtime'],
            info['start_time'],
            info['error'][:19] if info['error'] else '-'
        )
    
    # Create log panel with recent logs
    log_lines = list(log_queue)[-max_logs:]  # Get last N logs
    log_text = "\n".join(log_lines) if log_lines else "[dim]No logs yet...[/dim]"
    log_panel = Panel(
        log_text,
        title="📝 Activity Log",
        border_style="blue",
        padding=(1, 2)
    )

    # Create layout
    layout = Layout()
    layout.split_column(
        Layout(table, name="status", ratio=2),
        Layout(log_panel, name="logs", ratio=1)
    )

    return layout


def run_multiple_pods(pod_configs, API_KEY):
    """Run multiple pods in parallel with dashboard monitoring"""
    manager = Manager()
    status_dict = manager.dict()
    log_queue = manager.list()  # Shared log queue
    processes = []
    
    console = Console()
    console.print(f"[cyan]Initializing {len(pod_configs)} pod(s)...[/cyan]")

    # Start all processes
    for idx, pod_config in enumerate(pod_configs):
        console.print(f"[yellow]Starting pod {idx}: {pod_config.get('name', 'unnamed')}[/yellow]")
        p = Process(target=run_single_pod, args=(pod_config, status_dict, idx, API_KEY, log_queue))
        p.start()
        processes.append(p)
        time.sleep(0.5)  # Stagger pod creation
    
    console.print("[green]All pods started. Monitoring...[/green]\n")

    # Monitor with live dashboard - faster refresh for smooth spinner
    interrupted = False
    try:
        with Live(generate_dashboard(status_dict, log_queue), refresh_per_second=4, console=console) as live:
            while any(p.is_alive() for p in processes):
                live.update(generate_dashboard(status_dict, log_queue))
                time.sleep(0.25)

            # Final update
            live.update(generate_dashboard(status_dict, log_queue))
    except KeyboardInterrupt:
        interrupted = True
        console.print("\n[red]⚠️  Interrupted by user (Ctrl+C)[/red]")
        console.print("[yellow]Cleaning up active pods...[/yellow]")

    # Wait for all processes to complete
    for p in processes:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
    
    # Emergency cleanup if interrupted
    if interrupted:
        cleanup_all_pods()

    console.print("\n[green]✅ All pods completed![/green]\n")

    # Print final summary
    final_table = generate_dashboard(dict(status_dict), log_queue)
    console.print(final_table)

    # Generate summary statistics
    final_status = dict(status_dict)
    total = len(final_status)
    completed = sum(1 for s in final_status.values() if '✅ EXITED' in s['status'] or '🛑 TERMINATED' in s['status'])
    errors = sum(1 for s in final_status.values() if '❌ ERROR' in s['status'])
    timeouts = sum(1 for s in final_status.values() if '⏱️ TIMEOUT' in s['status'])

    console.print("\n" + "="*80)
    console.print("[bold cyan]📊 Summary Report[/bold cyan]")
    console.print("="*80)
    console.print(f"Total pods: [bold]{total}[/bold]")
    console.print(f"✅ Completed: [green]{completed}[/green]")
    console.print(f"❌ Errors: [red]{errors}[/red]")
    console.print(f"⏱️ Timeouts: [yellow]{timeouts}[/yellow]")
    console.print("="*80 + "\n")


def main(args):
    """Main entry point"""
    # Register cleanup handlers for emergency termination
    signal.signal(signal.SIGINT, cleanup_all_pods)   # Ctrl+C
    signal.signal(signal.SIGTERM, cleanup_all_pods)  # kill command
    atexit.register(cleanup_all_pods)  # Normal exit

    load_dotenv()
    API_KEY = os.getenv('RUNPOD_API_KEY')

    if not API_KEY:
        raise FileNotFoundError('Cannot access to API_KEY file. Please check .env file exists in current folder.')

    runpod.api_key = API_KEY

    pod_configs = []

    # Check if multiple configs provided
    if hasattr(args, 'configs') and args.configs:
        # Multi-pod mode
        for config_file in args.configs:
            config_path = Path(config_file)
            if not config_path.exists():
                print(f"Warning: Config file {config_file} not found, skipping...")
                continue

            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                pod_config = create_pod_config_from_yaml(config_data, args)
                pod_configs.append(pod_config)
    elif hasattr(args, 'config') and args.config:
        # Single config file - also use dashboard
        config_path = Path(config_folder_path) / args.config
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                pod_config = create_pod_config_from_yaml(config_data, args)
                pod_configs.append(pod_config)
        else:
            print(f"Warning: Config file {config_path} not found")

    if pod_configs:
        print(f"Starting {len(pod_configs)} pod(s) with dashboard...")
        run_multiple_pods(pod_configs, API_KEY)
    else:
        print("No valid pod configurations found.")




if __name__ == '__main__':
    args = parse_args()
    try:
        main(args)
    except Exception as e:
        _console.print(f"\n[bold red]💥 Fatal error: {str(e)}[/bold red]")
        _console.print("[yellow]Attempting emergency cleanup...[/yellow]")
        cleanup_all_pods()
        raise
    finally:
        # Final safety check - cleanup any remaining pods
        if _active_pods:
            _console.print("[yellow]⚠️  Cleaning up remaining pods...[/yellow]")
            cleanup_all_pods()

"""
Usage:

Single pod execution:
python runpod/deploy.py --config config.yaml

Multiple pods execution (multi-process with dashboard):
python runpod/deploy.py --configs config1.yaml config2.yaml config3.yaml

Direct arguments:
python runpod/deploy.py --name my-pod --gpu "NVIDIA RTX 2000 Ada Generation" --gpu-count 1
"""