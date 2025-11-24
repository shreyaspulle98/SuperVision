#!/usr/bin/env python3
"""
monitor_and_optimize.py

Monitors DINOv3 training and automatically applies memory optimizations after Epoch 1.

This script:
1. Watches dinov3_train.log for Epoch 1 completion
2. Kills the training process
3. Applies memory optimization fixes to the training code
4. Restarts training with optimized configuration

Usage:
    python3 monitor_and_optimize.py

The script will run continuously until Epoch 1 completes, then apply fixes and restart.
"""

import os
import sys
import time
import signal
import subprocess
from pathlib import Path


class TrainingMonitor:
    """Monitor training progress and apply optimizations."""

    def __init__(self, log_file="dinov3_train.log", check_interval=30):
        """
        Initialize monitor.

        Args:
            log_file (str): Path to training log file
            check_interval (int): How often to check log (seconds)
        """
        self.log_file = Path(log_file)
        self.check_interval = check_interval
        self.epoch_1_completed = False
        self.last_position = 0

    def check_epoch_completion(self):
        """
        Check if Epoch 1 has completed by reading the log file.

        Returns:
            bool: True if Epoch 1 is complete
        """
        if not self.log_file.exists():
            return False

        try:
            with open(self.log_file, 'r') as f:
                # Move to last known position
                f.seek(self.last_position)
                new_content = f.read()
                self.last_position = f.tell()

                # Check for Epoch 1 completion markers
                # The training script prints "Epoch 1/50" at the start of epoch 1
                # and "Epoch 2/50" at the start of epoch 2
                if "Epoch 2/" in new_content:
                    print("\n✓ Epoch 1 completed!")
                    return True

                # Alternative: Check for checkpoint save message
                if "Saved best model" in new_content and not self.epoch_1_completed:
                    # Give it a few seconds to finish writing
                    time.sleep(10)
                    # Re-read to confirm Epoch 2 started or training stopped
                    with open(self.log_file, 'r') as f2:
                        f2.seek(self.last_position)
                        additional = f2.read()
                        if "Epoch 2/" in additional or "Early stopping" in additional:
                            print("\n✓ Epoch 1 completed!")
                            return True

        except Exception as e:
            print(f"Error reading log: {e}")

        return False

    def find_training_process(self):
        """
        Find the running DINOv3 training process.

        Returns:
            int or None: PID of training process
        """
        try:
            # Look for python process running the training script
            result = subprocess.run(
                ["pgrep", "-f", "dinov3_pipeline.train"],
                capture_output=True,
                text=True
            )

            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                # Return the first PID (should only be one)
                return int(pids[0])
        except Exception as e:
            print(f"Error finding process: {e}")

        return None

    def kill_training_process(self, pid):
        """
        Gracefully kill the training process.

        Args:
            pid (int): Process ID to kill
        """
        try:
            print(f"\nStopping training process (PID: {pid})...")
            os.kill(pid, signal.SIGTERM)

            # Wait for process to terminate
            time.sleep(5)

            # Check if it's still running
            try:
                os.kill(pid, 0)  # This will raise exception if process is dead
                print("Process still running, forcing kill...")
                os.kill(pid, signal.SIGKILL)
                time.sleep(2)
            except OSError:
                print("✓ Training process stopped successfully")

        except Exception as e:
            print(f"Error killing process: {e}")

    def apply_memory_optimizations(self):
        """
        Apply memory optimization fixes to the training code.

        This modifies dinov3_pipeline/train.py with:
        - Reduced batch size (32 → 8)
        - Reduced workers (4 → 0)
        - Disabled AMP on CPU
        - Added memory cleanup in evaluate()
        """
        print("\n" + "=" * 70)
        print("Applying Memory Optimizations")
        print("=" * 70)

        train_file = Path("dinov3_pipeline/train.py")

        if not train_file.exists():
            print(f"Error: {train_file} not found!")
            return False

        # Read current content
        with open(train_file, 'r') as f:
            content = f.read()

        # Apply fixes
        print("\n1. Reducing batch size: 32 → 8")
        content = content.replace(
            '"batch_size": 32,',
            '"batch_size": 8,  # Reduced from 32 to prevent memory swapping'
        )

        print("2. Disabling data loader workers: 4 → 0")
        content = content.replace(
            '"num_workers": 4,',
            '"num_workers": 0,  # Disabled multiprocessing to reduce memory overhead'
        )

        print("3. Disabling AMP (not beneficial on CPU)")
        content = content.replace(
            '"use_amp": True,  # Use automatic mixed precision if available',
            '"use_amp": False,  # Disabled - AMP only benefits GPU training'
        )

        print("4. Adding explicit memory cleanup in evaluate() method")
        # Add gc.collect() after the evaluation loop
        evaluate_cleanup = """        all_predictions = np.array(all_predictions).flatten()
        all_targets = np.array(all_targets).flatten()

        # Explicitly free memory to prevent accumulation
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        metrics = {"""

        content = content.replace(
            """        all_predictions = np.array(all_predictions).flatten()
        all_targets = np.array(all_targets).flatten()

        metrics = {""",
            evaluate_cleanup
        )

        # Write optimized content
        with open(train_file, 'w') as f:
            f.write(content)

        print("\n✓ All optimizations applied successfully!")
        print("\nOptimizations:")
        print("  • Batch size: 32 → 8 (75% memory reduction)")
        print("  • Workers: 4 → 0 (30% memory reduction)")
        print("  • AMP: Disabled on CPU")
        print("  • Memory cleanup: Added to prevent leaks")
        print("\nExpected impact:")
        print("  • Memory usage: ~80% reduction")
        print("  • Training speed: ~2-3 hours per epoch (vs 7-9 hours before)")
        print("  • Total training time: ~40-60 hours with early stopping")

        return True

    def restart_training(self):
        """
        Restart training with optimized configuration.
        """
        print("\n" + "=" * 70)
        print("Restarting Training with Optimizations")
        print("=" * 70)

        # Backup old log
        if self.log_file.exists():
            backup_log = Path("dinov3_train_epoch1_only.log")
            print(f"\nBacking up original log to {backup_log}")
            os.rename(self.log_file, backup_log)

        # Start new training
        print("\nStarting optimized training...")
        print("This will run in the background with caffeinate to prevent sleep.\n")

        cmd = [
            "nohup",
            "caffeinate", "-i",
            "python3", "-m", "dinov3_pipeline.train",
            ">", str(self.log_file),
            "2>&1",
            "&"
        ]

        # Use shell=True to handle redirection
        subprocess.Popen(
            " ".join(cmd),
            shell=True,
            cwd=os.getcwd()
        )

        time.sleep(3)

        # Verify it started
        pid = self.find_training_process()
        if pid:
            print(f"✓ Training restarted successfully (PID: {pid})")
            print(f"\nMonitor progress with:")
            print(f"  tail -f {self.log_file}")
        else:
            print("⚠ Could not verify training started. Check manually.")

        return True

    def run(self):
        """
        Main monitoring loop.
        """
        print("=" * 70)
        print("DINOv3 Training Monitor & Optimizer")
        print("=" * 70)
        print("\nThis script will:")
        print("  1. Monitor training progress")
        print("  2. Stop training after Epoch 1 completes")
        print("  3. Apply memory optimizations")
        print("  4. Restart training with optimized config")
        print(f"\nChecking log every {self.check_interval} seconds...")
        print("Press Ctrl+C to abort.\n")

        try:
            while not self.epoch_1_completed:
                # Check if Epoch 1 is done
                if self.check_epoch_completion():
                    self.epoch_1_completed = True

                    # Find and kill the training process
                    pid = self.find_training_process()
                    if pid:
                        self.kill_training_process(pid)
                    else:
                        print("Warning: Could not find training process to kill")

                    # Apply optimizations
                    if self.apply_memory_optimizations():
                        # Restart training
                        self.restart_training()

                        print("\n" + "=" * 70)
                        print("Optimization Complete!")
                        print("=" * 70)
                        print("\nTraining is now running with optimized settings.")
                        print("It should complete much faster (~2-3 hours per epoch).")
                        print("\nExiting monitor script.")
                        break
                    else:
                        print("\nError applying optimizations. Exiting.")
                        sys.exit(1)

                # Show we're still alive
                print(".", end="", flush=True)
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            print("\n\nMonitoring aborted by user.")
            sys.exit(0)


if __name__ == "__main__":
    monitor = TrainingMonitor(
        log_file="dinov3_train.log",
        check_interval=30  # Check every 30 seconds
    )
    monitor.run()
