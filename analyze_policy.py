"""
analyze_policy.py

Visualizes policy distributions and opening preferences of the trained ChessNet.
Generates SVG/PNG images of the board with arrows indicating the model's preferred moves.

Usage:
    python analyze_policy.py --model_path checkpoint.pth --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
"""

import argparse
import torch
import chess
import chess.svg
import numpy as np
import matplotlib.pyplot as plt
import io
import cairosvg
from PIL import Image

from model import ChessNet
from encode import board_to_tensor, move_to_action

def load_model(path, device='cpu'):
    """Loads the trained model from a checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    
    # determine model architecture from checkpoint metadata or use defaults
    metadata = checkpoint.get("metadata", {})
    settings = metadata.get("settings", {})
    num_blocks = settings.get("model_residual_blocks", 6)
    num_channels = settings.get("model_channels", 128)
    
    # create model with correct architecture
    model = ChessNet(
        num_residual_blocks=num_blocks,
        num_channels=num_channels
    )
    
    # load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)  # Support loading raw state dict
    
    model.to(device)
    model.eval()
    
    print(f"Loaded model: {num_blocks} blocks, {num_channels} channels")
    return model

def get_policy_value(model, board):
    """Runs inference on a chess.Board."""
    device = next(model.parameters()).device
    tensor = board_to_tensor(board).unsqueeze(0).to(device)
    with torch.no_grad():
        policy_logits, value = model(tensor)
    
    return policy_logits.cpu(), value.item()

def visualize_policy(model, board, output_file="policy_viz.png", top_k=5):
    """
    Visualizes the top K moves from the policy on the board using arrows.
    """
    policy_logits, value = get_policy_value(model, board)
    
    # Decode policy
    legal_moves = list(board.legal_moves)
    move_probs = []
    
    for move in legal_moves:
        action_data = move_to_action(move, board)
        if action_data is None:
            continue
        from_sq, action_type = action_data
        r, c = from_sq // 8, from_sq % 8
        
        logit = policy_logits[0, r, c, action_type].item()
        move_probs.append((move, logit))
        
    # Softmax over legal moves
    if not move_probs:
        print("No legal moves found/encodable.")
        return

    moves, logits = zip(*move_probs)
    probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
    
    # Sort by probability
    sorted_indices = np.argsort(probs)[::-1]
    top_moves = []
    
    # Prepare arrows for SVG
    arrows = []
    print(f"Position Value Estimate: {value:.3f} (1=White win, -1=Black win)")
    print("Top Moves:")
    
    for i in range(min(top_k, len(moves))):
        idx = sorted_indices[i]
        move = moves[idx]
        prob = probs[idx]
        print(f"{i+1}. {board.san(move)}: {prob:.1%}")
        
        # Arrow opacity/color based on probability
        # Opacity is not directly supported in simple chess.svg arrow lists easily without custom CSS, 
        # but we can use color. Green -> likely, Blue -> less likely.
        color = f"#00ff00{int(prob*255):02x}" # Green with alpha
        
        # chess.svg.Arrow(tail, head, color)
        arrows.append(chess.svg.Arrow(move.from_square, move.to_square, color=color))

    # Render SVG
    svg_data = chess.svg.board(board, arrows=arrows, size=400)
    
    # Convert to PNG
    png_bytes = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
    image = Image.open(io.BytesIO(png_bytes))
    image.save(output_file)
    print(f"Saved visualization to {output_file}")

def analyze_openings(model):
    """Analyzes model preferences for standard opening positions."""
    openings = {
        "Starting Position": chess.STARTING_FEN,
        "e4 (King's Pawn)": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        "d4 (Queen's Pawn)": "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
        "Sicilian Defense": "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "French Defense": "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    }
    
    print("\n--- Opening Analysis ---")
    for name, fen in openings.items():
        print(f"\nAnalyzing: {name}")
        board = chess.Board(fen)
        visualize_policy(model, board, output_file=f"viz_{name.replace(' ', '_').lower()}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--fen", type=str, default=chess.STARTING_FEN, help="FEN string to analyze")
    parser.add_argument("--analyze_openings", action="store_true", help="Run analysis on standard openings")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/mps/cpu). Auto-detects if not specified.")
    
    args = parser.parse_args()
    
    # auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"  # Metal Performance Shaders for M1/M2 Macs
        else:
            device = "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    try:
        model = load_model(args.model_path, device=device)
        
        if args.analyze_openings:
            analyze_openings(model)
        else:
            board = chess.Board(args.fen)
            visualize_policy(model, board)
            
    except FileNotFoundError:
        print(f"Error: Model file '{args.model_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

