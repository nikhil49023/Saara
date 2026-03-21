"""
🪔 SAARA Splash Screen

A beautiful gradient ASCII art splash screen.
Optimized for clarity and minimal distraction.

© 2025-2026 Kilani Sai Nikhil. All Rights Reserved.
"""

import time

def display_animated_splash(duration: float = 2.5):
    """
    Display a clean, minimal splash screen with SAARA in yellow-orange gradient.
    
    Args:
        duration: How long to show the animation (seconds). Set to 0 for infinite.
    """
    try:
        from rich.console import Console
        from rich.text import Text
        from rich.panel import Panel
        from rich.align import Align
        
        console = Console()
        
        # ANSI color codes for gradient (yellow -> orange)
        # Using 256-color palette for smooth gradient
        gradient_colors = [
            '\033[38;5;226m',  # Bright Yellow
            '\033[38;5;220m',  # Gold
            '\033[38;5;214m',  # Orange-Yellow
            '\033[38;5;208m',  # Orange
            '\033[38;5;202m',  # Dark Orange
        ]
        
        C_GREY = '\033[90m'
        C_WHITE = '\033[97m'
        RESET = '\033[0m'
        BOLD = '\033[1m'
        
        # Cursor controls
        HIDE_CURSOR = '\033[?25l'
        SHOW_CURSOR = '\033[?25h'
        
        # Big ASCII Art for SAARA with gradient
        SAARA_ART = [
            "  ███████╗  █████╗   █████╗  ██████╗   █████╗  ",
            "  ██╔════╝ ██╔══██╗ ██╔══██╗ ██╔══██╗ ██╔══██╗ ",
            "  ███████╗ ███████║ ███████║ ██████╔╝ ███████║ ",
            "  ╚════██║ ██╔══██║ ██╔══██║ ██╔══██╗ ██╔══██║ ",
            "  ███████║ ██║  ██║ ██║  ██║ ██║  ██║ ██║  ██║ ",
            "  ╚══════╝ ╚═╝  ╚═╝ ╚═╝  ╚═╝ ╚═╝  ╚═╝ ╚═╝  ╚═╝ ",
        ]
        
        def get_gradient_line(line: str, frame_offset: int = 0) -> str:
            """Apply gradient colors to a line."""
            result = ""
            chars = list(line)
            total_chars = len(chars)
            
            for i, char in enumerate(chars):
                # Calculate color index based on position + frame offset
                color_idx = (i + frame_offset) % len(gradient_colors)
                result += gradient_colors[color_idx] + char
            
            return result + RESET
        
        def get_frame(frame_idx: int) -> str:
            """Generate a single frame with animated gradient."""
            lines = ["\n"]
            
            # Add gradient SAARA art
            for art_line in SAARA_ART:
                lines.append("    " + get_gradient_line(art_line, frame_idx))
            
            lines.append("")
            lines.append(f"    {C_GREY}────────────────────────────────────────────────────{RESET}")
            lines.append(f"    {BOLD}{C_WHITE}Autonomous Document-to-LLM Data Engine{RESET}")
            lines.append(f"    {C_GREY}© 2025-2026 Kilani Sai Nikhil. All Rights Reserved.{RESET}")
            lines.append("")
            
            return '\n'.join(lines)
        
        # Run animation
        print(HIDE_CURSOR, end='', flush=True)
        
        # Get frame dimensions
        first_frame = get_frame(0)
        height = first_frame.count('\n') + 1
        
        # Print first frame
        print(first_frame)
        
        start_time = time.time()
        idx = 0
        
        while True:
            if duration > 0 and (time.time() - start_time) >= duration:
                break
            
            time.sleep(0.1)
            idx += 1
            
            # Move cursor UP
            print(f'\033[{height}A', end='')
            
            # Print next frame
            print(get_frame(idx))
            
    except KeyboardInterrupt:
        pass
    except Exception:
        # Fallback
        display_splash(animate=False)
    finally:
        print(SHOW_CURSOR, end='', flush=True)
        print()


def display_splash(animate: bool = True):
    """Static fallback for splash screen."""
    # Gradient colors
    C_YELLOW = '\033[38;5;226m'
    C_GOLD = '\033[38;5;220m'
    C_ORANGE = '\033[38;5;208m'
    C_WHITE = '\033[97m'
    C_GREY = '\033[90m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    print()
    print(f"    {C_YELLOW}  ███████╗{C_GOLD}  █████╗ {C_GOLD}  █████╗ {C_ORANGE} ██████╗ {C_ORANGE}  █████╗  {RESET}")
    print(f"    {C_YELLOW}  ██╔════╝{C_GOLD} ██╔══██╗{C_GOLD} ██╔══██╗{C_ORANGE} ██╔══██╗{C_ORANGE} ██╔══██╗ {RESET}")
    print(f"    {C_YELLOW}  ███████╗{C_GOLD} ███████║{C_GOLD} ███████║{C_ORANGE} ██████╔╝{C_ORANGE} ███████║ {RESET}")
    print(f"    {C_YELLOW}  ╚════██║{C_GOLD} ██╔══██║{C_GOLD} ██╔══██║{C_ORANGE} ██╔══██╗{C_ORANGE} ██╔══██║ {RESET}")
    print(f"    {C_YELLOW}  ███████║{C_GOLD} ██║  ██║{C_GOLD} ██║  ██║{C_ORANGE} ██║  ██║{C_ORANGE} ██║  ██║ {RESET}")
    print(f"    {C_YELLOW}  ╚══════╝{C_GOLD} ╚═╝  ╚═╝{C_GOLD} ╚═╝  ╚═╝{C_ORANGE} ╚═╝  ╚═╝{C_ORANGE} ╚═╝  ╚═╝ {RESET}")
    print()
    print(f"    {C_GREY}────────────────────────────────────────────────────{RESET}")
    print(f"    {BOLD}{C_WHITE}Autonomous Document-to-LLM Data Engine{RESET}")
    print(f"    {C_GREY}© 2025-2026 Kilani Sai Nikhil. All Rights Reserved.{RESET}")
    print()


def display_minimal_header():
    """Display a compact single-line header."""
    C_GOLD = '\033[38;5;220m'
    C_ORANGE = '\033[38;5;208m'
    C_GREY = '\033[90m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Gradient text for minimal header
    print(f"\n{C_GOLD}S{C_GOLD}A{C_ORANGE}A{C_ORANGE}R{C_ORANGE}A{RESET} {C_GREY}• Autonomous LLM Data Engine{RESET}")
    print(f"{C_GREY}{'─' * 40}{RESET}\n")


def display_version():
    """Display version information."""
    from importlib.metadata import version as get_version
    try:
        ver = get_version("saara-ai")
    except:
        ver = "dev"
    
    C_GOLD = '\033[38;5;220m'
    C_ORANGE = '\033[38;5;208m'
    C_GREY = '\033[90m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
        
    print(f"\n{BOLD}{C_GOLD}SAARA{RESET} v{ver}")
    print(f"{C_GREY}Autonomous Document-to-LLM Data Engine{RESET}")
    print(f"{C_GREY}© 2025-2026 Kilani Sai Nikhil. All Rights Reserved.{RESET}\n")


def display_goodbye():
    """Display goodbye message."""
    C_GOLD = '\033[38;5;220m'
    C_GREY = '\033[90m'
    RESET = '\033[0m'
    
    print(f"\n{C_GOLD}SAARA{RESET} {C_GREY}• Thank you for using SAARA!{RESET}\n")


if __name__ == "__main__":
    display_animated_splash(duration=5.0)
