import xml.etree.ElementTree as ET
import json
import argparse
import os
import numpy as np

def parse_barn_world(input_file, output_file):
    try:
        tree = ET.parse(input_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return

    # Initialize the dictionary structure
    data = {
        "sudden": [],   # Red inner obstacles
        "visible": []   # Blue boundary walls
    }

    # Find the <world> tag. SDF files usually have <sdf><world>...
    world = root.find("world")
    if world is None:
        print("Error: Could not find <world> tag inside SDF.")
        return

    # Iterate through all models in the world
    for model in world.findall("model"):
        name = model.get("name")
        
        # Skip ground plane or other non-cylinder models
        if "cylinder" not in name:
            continue

        try:
            # 1. Get Position (x, y)
            pose_str = model.find("pose").text
            pose_values = [float(x) for x in pose_str.split()]
            x_center = pose_values[0]
            y_center = pose_values[1]

            # 2. Get Radius
            # Path: link -> collision -> geometry -> cylinder -> radius
            link = model.find("link")
            collision = link.find("collision")
            geometry = collision.find("geometry")
            cylinder = geometry.find("cylinder")
            radius = float(cylinder.find("radius").text)

            # 3. Get Color to determine Type
            # Path: link -> visual -> material -> diffuse (or ambient)
            visual = link.find("visual")
            material = visual.find("material")
            
            # The files use 'diffuse' or 'ambient' for color. 
            # Format is usually "R G B A" string
            color_elem = material.find("diffuse")
            if color_elem is None:
                color_elem = material.find("ambient")
            
            color_str = color_elem.text
            r, g, b, _ = [float(c) for c in color_str.split()]

            # 4. Classification Logic
            entry = [x_center, y_center, radius]

            # In this dataset:
            # Red (High R, Low B) = Sudden/Obstacle
            # Blue (Low R, High B) = Visible/Boundary
            if r > b:
                data["sudden"].append(entry)
            else:
                data["visible"].append(entry)

        except AttributeError:
            # Skip models that represent cylinders but lack specific tags (malformed)
            continue

    # Write to NPZ
    # Arrays will have shape (N, 3) containing [x, y, radius]
    np.savez(
        output_file, 
        sudden=np.array(data["sudden"]), 
        visible=np.array(data["visible"])
    )
    
    print(f"Successfully parsed stats:")
    print(f" - Visible (Boundary) Cylinders: {len(data['visible'])}")
    print(f" - Sudden (Inner) Cylinders: {len(data['sudden'])}")
    print(f" - Saved to: {output_file}")

# --- Execution Example ---
if __name__ == "__main__":
    # Example usage:
    # python scripts/eval/parse_barn.py --input_dir ./worlds --output_dir ./parsed_maps
    
    # How to read the output .npz file:
    # ------------------------------------------------
    # import numpy as np
    # data = np.load('world_0.npz')
    # sudden_obs = data['sudden']   # shape (N, 3) -> [x, y, radius]
    # visible_obs = data['visible'] # shape (M, 3) -> [x, y, radius]
    # ------------------------------------------------

    parser = argparse.ArgumentParser(description="Parse BARN world files to NPZ.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .world files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save .npz files")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for filename in os.listdir(args.input_dir):
        if filename.endswith(".world"):
            input_path = os.path.join(args.input_dir, filename)
            output_filename = filename.replace(".world", ".npz")
            output_path = os.path.join(args.output_dir, output_filename)
            
            print(f"Processing {filename}...")
            parse_barn_world(input_path, output_path)
