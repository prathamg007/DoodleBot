import socket
import sys
import time

# --- Configuration ---
# IP address of your Raspberry Pi 2 (the motor control RPi)
# REPLACE WITH YOUR RPI 2's ACTUAL IP ADDRESS!
RPI2_IP = "192.168.50.195" 
RPI2_PORT = 8888 # Must match the port RPi 2 is listening on

def send_command(sock, command):
    """Sends a UDP command to the RPi 2."""
    try:
        sock.sendto(command.encode(), (RPI2_IP, RPI2_PORT))
        print(f"Sent: '{command}' to {RPI2_IP}:{RPI2_PORT}")
    except Exception as e:
        print(f"Error sending command: {e}")

def main():
    print("DoodleBot PC Client - Manual Motor Control")
    print(f"Connecting to Raspberry Pi 2 at {RPI2_IP}:{RPI2_PORT}")
    print("\nAvailable Commands:")
    print("  'w'   - Move both motors forward (speed 50)")
    print("  's'   - Stop both motors")
    print("  'a'   - Turn left (Left motor backward, Right motor forward)")
    print("  'd'   - Turn right (Left motor forward, Right motor backward)")
    print("  'q'   - Quit")
    print("  'L<speed>R<speed>' - Direct control (e.g., L100R-50 for left full forward, right half backward)")
    print("  (Speed values range from -100 to 100)")
    
    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    try:
        while True:
            cmd_input = input("\nEnter command: ").strip().lower()
            
            if cmd_input == 'q':
                print("Quitting.")
                send_command(sock, "s") # Send a stop command before exiting
                break
            elif cmd_input == 'w':
                send_command(sock, "L50R50") # Both forward at 50%
            elif cmd_input == 's':
                send_command(sock, "s") # Stop
            elif cmd_input == 'a':
                send_command(sock, "L-30R30") # Turn left (adjust speeds as needed)
            elif cmd_input == 'd':
                send_command(sock, "L30R-30") # Turn right (adjust speeds as needed)
            elif cmd_input.startswith("l") and "r" in cmd_input:
                # Direct speed command (e.g., L100R-50)
                # Ensure it's correctly formatted to pass through parsing on RPi
                try:
                    parts = cmd_input.upper().split('R')
                    left_speed = int(parts[0][1:])
                    right_speed = int(parts[1])
                    
                    # Clamp speeds to -100 to 100
                    left_speed = max(-100, min(100, left_speed))
                    right_speed = max(-100, min(100, right_speed))
                    
                    send_command(sock, f"L{left_speed}R{right_speed}")
                except ValueError:
                    print("Invalid speed format. Use L<int>R<int> (e.g., L50R-20)")
                except IndexError:
                    print("Malformed command. Use L<speed>R<speed>.")
            else:
                print("Invalid command. Please use 'w', 's', 'a', 'd', 'q', or 'L<speed>R<speed>'.")
            
            time.sleep(0.1) # Small delay to avoid flooding the network

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C).")
    finally:
        send_command(sock, "s") # Ensure motors stop on RPi 2 if client exits unexpectedly
        sock.close()
        print("PC client cleanup complete.")

if _name_ == "_main_":
    main()