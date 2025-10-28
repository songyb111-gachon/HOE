"""
PyTorch Trainer for HOE Models
Flexible training loop with logging, checkpointing, and visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
from datetime import datetime
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures


class Trainer:
    """General purpose trainer for HOE models"""
    
    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 criterion,
                 optimizer,
                 device='cuda',
                 checkpoint_dir='checkpoints',
                 log_dir='logs',
                 experiment_name=None,
                 visualize_freq=10):
        """
        Args:
            model: PyTorch model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            criterion: Loss function
            optimizer: Optimizer
            device: 'cuda' or 'cpu'
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for TensorBoard logs
            experiment_name: Name of this experiment
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
        # Setup directories
        if experiment_name is None:
            experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_name = experiment_name
        
        self.checkpoint_dir = Path(checkpoint_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(str(self.log_dir))
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.train_mse = []
        self.val_mse = []
        self.train_psnr = []
        self.val_psnr = []
        self.visualize_freq = visualize_freq
        
        # Create visualization directory
        self.viz_dir = self.checkpoint_dir / 'visualizations'
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"Trainer Initialized")
        print(f"{'='*80}")
        print(f"  • Experiment: {self.experiment_name}")
        print(f"  • Device: {self.device}")
        print(f"  • Checkpoint dir: {self.checkpoint_dir}")
        print(f"  • Log dir: {self.log_dir}")
        print(f"  • Visualization dir: {self.viz_dir}")
        print(f"  • Train samples: {len(train_loader.dataset)}")
        print(f"  • Val samples: {len(val_loader.dataset)}")
        print(f"  • Visualization frequency: every {visualize_freq} epochs")
        print(f"{'='*80}\n")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1} [Train]')
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(self.device)
            
            # Handle different target formats
            if 'target' in batch:
                targets = batch['target'].to(self.device)
            elif 'targets' in batch:
                targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
            else:
                raise ValueError("Batch must contain 'target' or 'targets'")
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate loss
            if isinstance(targets, dict):
                # Multi-output case
                if isinstance(self.criterion, nn.ModuleList):
                    # Separate loss for each output
                    loss = 0
                    for i, (output, (key, target)) in enumerate(zip(outputs, targets.items())):
                        loss += self.criterion[i](output, target)
                else:
                    # Single criterion handles multiple outputs
                    loss, _ = self.criterion(outputs, list(targets.values()))
            else:
                # Single output case
                loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate MSE for monitoring
            with torch.no_grad():
                if isinstance(targets, dict):
                    # For multi-output, calculate MSE on first output
                    target_for_mse = list(targets.values())[0]
                    output_for_mse = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                else:
                    target_for_mse = targets
                    output_for_mse = outputs
                
                # Apply sigmoid if using BCEWithLogitsLoss
                if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                    output_for_mse = torch.sigmoid(output_for_mse)
                
                mse = nn.functional.mse_loss(output_for_mse, target_for_mse)
                epoch_mse += mse.item()
            
            # Update progress
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}', 'mse': f'{mse.item():.6f}'})
        
        avg_loss = epoch_loss / len(self.train_loader)
        avg_mse = epoch_mse / len(self.train_loader)
        
        # Calculate PSNR from MSE (assuming data range [0, 1])
        avg_psnr = 10 * np.log10(1.0 / (avg_mse + 1e-10))
        
        return avg_loss, avg_mse, avg_psnr
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        epoch_loss = 0.0
        epoch_mse = 0.0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch+1} [Val]')
            for batch in pbar:
                # Move data to device
                images = batch['image'].to(self.device)
                
                # Handle different target formats
                if 'target' in batch:
                    targets = batch['target'].to(self.device)
                elif 'targets' in batch:
                    targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
                else:
                    raise ValueError("Batch must contain 'target' or 'targets'")
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                if isinstance(targets, dict):
                    if isinstance(self.criterion, nn.ModuleList):
                        loss = 0
                        for i, (output, (key, target)) in enumerate(zip(outputs, targets.items())):
                            loss += self.criterion[i](output, target)
                    else:
                        loss, _ = self.criterion(outputs, list(targets.values()))
                else:
                    loss = self.criterion(outputs, targets)
                
                # Calculate MSE for monitoring
                if isinstance(targets, dict):
                    # For multi-output, calculate MSE on first output
                    target_for_mse = list(targets.values())[0]
                    output_for_mse = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                else:
                    target_for_mse = targets
                    output_for_mse = outputs
                
                # Apply sigmoid if using BCEWithLogitsLoss
                if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                    output_for_mse = torch.sigmoid(output_for_mse)
                
                mse = nn.functional.mse_loss(output_for_mse, target_for_mse)
                epoch_mse += mse.item()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.6f}', 'mse': f'{mse.item():.6f}'})
        
        avg_loss = epoch_loss / len(self.val_loader)
        avg_mse = epoch_mse / len(self.val_loader)
        
        # Calculate PSNR from MSE (assuming data range [0, 1])
        avg_psnr = 10 * np.log10(1.0 / (avg_mse + 1e-10))
        
        return avg_loss, avg_mse, avg_psnr
    
    def save_checkpoint(self, filename='checkpoint.pth', is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_mse': self.train_mse,
            'val_mse': self.val_mse,
            'train_psnr': self.train_psnr,
            'val_psnr': self.val_psnr,
            'best_val_loss': self.best_val_loss,
        }
        
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"  💾 Saved best model (val_loss: {self.best_val_loss:.6f})")
    
    def load_checkpoint(self, filename='checkpoint.pth'):
        """Load checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename
        
        if not checkpoint_path.exists():
            print(f"  ⚠️ Checkpoint not found: {checkpoint_path}")
            return False
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_mse = checkpoint.get('train_mse', [])
        self.val_mse = checkpoint.get('val_mse', [])
        self.train_psnr = checkpoint.get('train_psnr', [])
        self.val_psnr = checkpoint.get('val_psnr', [])
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"  ✓ Loaded checkpoint from epoch {self.current_epoch}")
        return True
    
    def visualize_predictions(self, num_samples=4):
        """Visualize validation predictions and save to file"""
        self.model.eval()
        
        with torch.no_grad():
            # Get one batch from validation
            val_batch = next(iter(self.val_loader))
            images = val_batch['image'].to(self.device)
            targets = val_batch['target'].to(self.device)
            
            # Predict
            outputs = self.model(images)
            
            # Apply sigmoid if using BCEWithLogitsLoss
            if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                outputs = torch.sigmoid(outputs)
            
            # Move to CPU
            images = images.cpu().numpy()
            targets = targets.cpu().numpy()
            outputs = outputs.cpu().numpy()
            
            # Plot
            num_samples = min(num_samples, images.shape[0])
            fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
            
            if num_samples == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(num_samples):
                # Input
                axes[i, 0].imshow(images[i, 0], cmap='viridis')
                axes[i, 0].set_title(f'Input')
                axes[i, 0].axis('off')
                
                # Ground Truth
                axes[i, 1].imshow(targets[i, 0], cmap='gray')
                axes[i, 1].set_title(f'Ground Truth')
                axes[i, 1].axis('off')
                
                # Prediction
                axes[i, 2].imshow(outputs[i, 0], cmap='gray')
                mse = np.mean((targets[i, 0] - outputs[i, 0])**2)
                psnr = 10 * np.log10(1.0 / (mse + 1e-10))
                axes[i, 2].set_title(f'Prediction\nMSE: {mse:.4f} | PSNR: {psnr:.2f} dB')
                axes[i, 2].axis('off')
            
            plt.suptitle(f'Epoch {self.current_epoch + 1} - Validation Predictions', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save figure
            save_path = self.viz_dir / f'predictions_epoch_{self.current_epoch+1:03d}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  📊 Saved predictions: {save_path.name}")
    
    def train(self, num_epochs, save_freq=5):
        """Train for multiple epochs
        
        Args:
            num_epochs: Number of epochs to train
            save_freq: Save checkpoint every N epochs
        """
        print(f"\n{'='*80}")
        print(f"Starting Training")
        print(f"{'='*80}")
        print(f"  • Total epochs: {num_epochs}")
        print(f"  • Save frequency: every {save_freq} epochs")
        print(f"  • Device: {self.device}")
        print(f"{'='*80}\n")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_mse, train_psnr = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_mse.append(train_mse)
            self.train_psnr.append(train_psnr)
            
            # Validate
            val_loss, val_mse, val_psnr = self.validate()
            self.val_losses.append(val_loss)
            self.val_mse.append(val_mse)
            self.val_psnr.append(val_psnr)
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('MSE/train', train_mse, epoch)
            self.writer.add_scalar('MSE/val', val_mse, epoch)
            self.writer.add_scalar('PSNR/train', train_psnr, epoch)
            self.writer.add_scalar('PSNR/val', val_psnr, epoch)
            self.writer.add_scalar('Learning_rate', 
                                  self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  • Train loss: {train_loss:.6f}  |  Train MSE: {train_mse:.6f}  |  Train PSNR: {train_psnr:.2f} dB")
            print(f"  • Val loss: {val_loss:.6f}  |  Val MSE: {val_mse:.6f}  |  Val PSNR: {val_psnr:.2f} dB")
            
            # Visualize predictions
            if (epoch + 1) % self.visualize_freq == 0 or (epoch + 1) == num_epochs:
                self.visualize_predictions(num_samples=4)
            
            # Save checkpoint
            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth', is_best=True)
            
            # Save latest checkpoint
            self.save_checkpoint('latest.pth')
        
        print(f"\n{'='*80}")
        print(f"Training Complete!")
        print(f"{'='*80}")
        print(f"  • Best val loss: {self.best_val_loss:.6f}")
        print(f"  • Final train loss: {self.train_losses[-1]:.6f}")
        print(f"  • Final val loss: {self.val_losses[-1]:.6f}")
        print(f"{'='*80}\n")
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_mse': self.train_mse,
            'val_mse': self.val_mse,
            'train_psnr': self.train_psnr,
            'val_psnr': self.val_psnr,
            'best_val_loss': self.best_val_loss,
        }
        
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        self.writer.close()


class MultiTaskTrainer(Trainer):
    """Trainer for multi-task models with separate losses per task"""
    
    def __init__(self, *args, task_names=None, **kwargs):
        """
        Args:
            *args: Arguments for base Trainer
            task_names: List of task names for logging
            **kwargs: Keyword arguments for base Trainer
        """
        super().__init__(*args, **kwargs)
        self.task_names = task_names or [f'Task_{i+1}' for i in range(len(self.criterion.task_losses))]
        self.task_train_losses = {name: [] for name in self.task_names}
        self.task_val_losses = {name: [] for name in self.task_names}
    
    def train_epoch(self):
        """Train for one epoch with per-task loss tracking"""
        self.model.train()
        epoch_loss = 0.0
        task_losses_sum = {name: 0.0 for name in self.task_names}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1} [Train]')
        for batch in pbar:
            images = batch['image'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Multi-task loss
            loss, task_losses_dict = self.criterion(outputs, list(targets.values()))
            
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            for i, name in enumerate(self.task_names):
                task_losses_sum[name] += task_losses_dict[f'task_{i+1}_loss']
            
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = epoch_loss / len(self.train_loader)
        avg_task_losses = {name: loss / len(self.train_loader) 
                          for name, loss in task_losses_sum.items()}
        
        return avg_loss, avg_task_losses
    
    def validate(self):
        """Validate with per-task loss tracking"""
        self.model.eval()
        epoch_loss = 0.0
        task_losses_sum = {name: 0.0 for name in self.task_names}
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch+1} [Val]')
            for batch in pbar:
                images = batch['image'].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
                
                outputs = self.model(images)
                loss, task_losses_dict = self.criterion(outputs, list(targets.values()))
                
                epoch_loss += loss.item()
                for i, name in enumerate(self.task_names):
                    task_losses_sum[name] += task_losses_dict[f'task_{i+1}_loss']
                
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = epoch_loss / len(self.val_loader)
        avg_task_losses = {name: loss / len(self.val_loader) 
                          for name, loss in task_losses_sum.items()}
        
        return avg_loss, avg_task_losses
    
    def train(self, num_epochs, save_freq=5):
        """Train with per-task logging"""
        print(f"\n{'='*80}")
        print(f"Starting Multi-Task Training")
        print(f"{'='*80}")
        print(f"  • Tasks: {', '.join(self.task_names)}")
        print(f"  • Total epochs: {num_epochs}")
        print(f"{'='*80}\n")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_task_losses = self.train_epoch()
            self.train_losses.append(train_loss)
            for name, loss in train_task_losses.items():
                self.task_train_losses[name].append(loss)
            
            # Validate
            val_loss, val_task_losses = self.validate()
            self.val_losses.append(val_loss)
            for name, loss in val_task_losses.items():
                self.task_val_losses[name].append(loss)
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train_total', train_loss, epoch)
            self.writer.add_scalar('Loss/val_total', val_loss, epoch)
            
            for name, loss in train_task_losses.items():
                self.writer.add_scalar(f'Loss_Task/train_{name}', loss, epoch)
            for name, loss in val_task_losses.items():
                self.writer.add_scalar(f'Loss_Task/val_{name}', loss, epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  • Total train loss: {train_loss:.6f}")
            print(f"  • Total val loss: {val_loss:.6f}")
            for name in self.task_names:
                print(f"  • {name}: train={train_task_losses[name]:.6f}, "
                      f"val={val_task_losses[name]:.6f}")
            
            # Save checkpoints
            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth', is_best=True)
            
            self.save_checkpoint('latest.pth')
        
        print(f"\n{'='*80}")
        print(f"Multi-Task Training Complete!")
        print(f"{'='*80}\n")
        
        self.writer.close()


if __name__ == "__main__":
    print("\n=== Trainer classes created successfully ===\n")
    print("✓ Trainer module ready!")

