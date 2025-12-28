"""
🪔 SAARA Splash Screen

A beautiful Sanskrit-inspired ASCII art splash with subtle animated "Lamp of Knowledge".
Optimized for clarity and minimal distraction.

© 2024-2025 Kilani Sai Nikhil. All Rights Reserved.
"""

import time
import random

def display_animated_splash(duration: float = 2.5):
    """
    Display a clean, minimal splash screen with explicit cursor control for robust in-place updates.
    
    Args:
        duration: How long to show the animation (seconds). Set to 0 for infinite.
    """
    try:
        from rich.console import Console
        
        console = Console()
        
        # Consistent Gold/Orange Palette (ANSI codes for manual print)
        C_FLAME = '\033[38;5;220m'  # Gold
        C_CORE = '\033[38;5;208m'   # Orange
        C_BASE = '\033[38;5;130m'   # Brown
        C_GOLD_MET = '\033[38;5;178m'
        C_CYAN = '\033[38;5;45m'
        C_WHITE = '\033[97m'
        C_GREY = '\033[90m'
        RESET = '\033[0m'
        BOLD = '\033[1m'
        
        # Cursor controls
        HIDE_CURSOR = '\033[?25l'
        SHOW_CURSOR = '\033[?25h'
        
        # Simplified Flame Frames
        FLAME_FRAMES = [
            ("   (  )   ", "  (    )  ", "   ) ॐ (  "),
            ("    )(    ", "  (    )  ", "   ) ॐ (  "), 
            ("   (  )   ", "   )  (   ", "   ) ॐ (  "),
            ("    )(    ", "   )  (   ", "   ) ॐ (  "),
        ]
        
        def get_frame_lines(frame_idx: int):
            frame = FLAME_FRAMES[frame_idx % len(FLAME_FRAMES)]
            
            # Use explicit spaces for centering (Line length ~ 50 chars)
            # Lamp indent ~ 20 
            I = "                    "
            
            lines = [
                "",
                f"{I}{C_GOLD_MET}{frame[0]}{RESET}",
                f"{I}{C_GOLD_MET}{frame[1]}{RESET}",
                f"{I}{C_CORE}{frame[2]}{RESET}",
                f"{I}{C_BASE} _||_   {RESET}",
                f"{I}{C_BASE}[████]  {RESET}",
                "",
                f"    {C_GOLD_MET} ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬       ▬▬▬▬▬▬▬▬▬▬{RESET}",
                f"    {C_CYAN} │  ╭──────────╮  │       │  ╭──────╯{RESET}",
                f"    {C_CYAN} │  ╰────╮     │  │       │  │       {RESET}",
                f"    {C_CYAN} ╰──────┼─────╯  │       │  ╰──────╮{RESET}",
                f"    {C_CYAN}        │        │       │         │{RESET}",
                "",
                f"         {BOLD}{C_WHITE}S  A  A  R  A{RESET}  {C_GREY}│{RESET}  {C_GOLD_MET}ज्ञानस्य सारः{RESET}",
                f"    {C_GREY}────────────────────────────────────────────{RESET}",
                f"    {BOLD}Autonomous Document-to-LLM Data Engine{RESET}",
                f"    {C_GREY}© 2024-2025 Kilani Sai Nikhil. All Rights Reserved.{RESET}",
                ""
            ]
            return lines

        # Run animation
        print(HIDE_CURSOR, end='', flush=True)
        
        # Pre-calculate height
        lines = get_frame_lines(0)
        height = len(lines)
        
        # Print first frame
        print('\n'.join(lines))
        
        start_time = time.time()
        idx = 0
        
        while True:
            if duration > 0 and (time.time() - start_time) >= duration:
                break
            
            time.sleep(0.15)
            idx += 1
            
            # Move cursor UP by 'height' lines
            print(f'\033[{height}A', end='')
            
            # Print next frame over it
            print('\n'.join(get_frame_lines(idx)))
            
    except KeyboardInterrupt:
        pass
    except Exception:
        # Fallback
        display_splash(animate=False)
    finally:
        print(SHOW_CURSOR, end='', flush=True)
        # Ensure we end on a new line
        print()


def display_splash(animate: bool = True):
    """Static fallback for splash screen."""
    C_GOLD = '\033[93m'
    C_CYAN = '\033[96m' 
    C_WHITE = '\033[97m'
    C_GREY = '\033[90m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    C_ORANGE = '\033[38;5;208m'
    
    # Static Print
    print(f"\n                    {C_GOLD}   (  )   {RESET}")
    print(f"                    {C_GOLD}  (    )  {RESET}")
    print(f"                    {C_GOLD}   ) ॐ (  {RESET}")
    print(f"                    {C_ORANGE}   _||_   {RESET}")
    print(f"                    {C_ORANGE}  [████]  {RESET}")
    print("")
    print(f"    {C_GOLD}  ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬       ▬▬▬▬▬▬▬▬▬▬{RESET}")
    print(f"    {C_CYAN}  │  ╭──────────╮  │       │  ╭──────╯{RESET}")
    print(f"    {C_CYAN}  │  ╰────╮     │  │       │  │       {RESET}")
    print(f"    {C_CYAN}  ╰──────┼─────╯  │       │  ╰──────╮{RESET}")
    print(f"    {C_CYAN}         │        │       │         │{RESET}")
    print("")
    print(f"         {BOLD}{C_WHITE}S  A  A  R  A{RESET}  {C_GREY}│{RESET}  {C_GOLD}ज्ञानस्य सारः{RESET}")
    print(f"    {C_GREY}────────────────────────────────────────────{RESET}")
    print(f"    {BOLD}Autonomous Document-to-LLM Data Engine{RESET}")
    print(f"    {C_GREY}© 2024-2025 Kilani Sai Nikhil. All Rights Reserved.{RESET}")
    print()

def display_minimal_header():
    """Display a compact single-line header."""
    C_GOLD = '\033[93m'
    C_CYAN = '\033[96m'
    C_GREY = '\033[90m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    print(f"\n{C_GOLD}🪔{RESET} {BOLD}{C_CYAN}SAARA{RESET} {C_GREY}• ज्ञानस्य सारः{RESET}")
    print(f"{C_GREY}{'─' * 40}{RESET}\n")

def display_version():
    """Display version information."""
    from importlib.metadata import version as get_version
    try:
        ver = get_version("saara-ai")
    except:
        ver = "dev"
    
    C_GOLD = '\033[93m'
    C_GREY = '\033[90m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
        
    print(f"\n{C_GOLD}🪔{RESET} {BOLD}SAARA{RESET} v{ver}")
    print(f"{C_GREY}The Essence of Knowledge • ज्ञानस्य सारः{RESET}")
    print(f"{C_GREY}© 2024-2025 Kilani Sai Nikhil. All Rights Reserved.{RESET}\n")

def display_goodbye():
    """Display goodbye message."""
    C_GOLD = '\033[93m'
    C_GREY = '\033[90m'
    RESET = '\033[0m'
    print(f"\n{C_GOLD}🪔{RESET} {C_GREY}May knowledge light your path. नमस्ते।{RESET}\n")

if __name__ == "__main__":
    display_animated_splash(duration=5.0)
