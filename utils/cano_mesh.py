import trimesh
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count


def normalize_mesh_to_unit_sphere(mesh):
    """
    Normalize mesh to unit sphere
    """
    vertices = mesh.vertices.copy()
    center = vertices.mean(axis=0)
    vertices -= center
    scale = (vertices**2).sum(axis=1).max()**0.5
    vertices /= scale
    mesh.vertices = vertices
    return mesh


def load_canoverse_data(json_path):
    """
    Load canoverse JSON file containing object category and rotation matrix information
    
    Args:
        json_path: Path to canoverse JSON file
        
    Returns:
        dict: Dictionary containing object information, format: {obj_id: {"category": str, "rotation_matrix": np.array (optional)}}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Check data format and process
    processed_data = {}
    for obj_id, obj_info in data.items():
            processed_obj = {
                "category": obj_info.get("category", "unknown")
            }
            
            # Check if rotation matrix is included
            if "rotation_matrix" in obj_info:
                rotation_matrix = obj_info["rotation_matrix"]
                if isinstance(rotation_matrix, list):
                    # Convert list to numpy array
                    rotation_matrix = np.array(rotation_matrix)
                    if rotation_matrix.shape == (3, 3):
                        # 3x3 rotation matrix, convert to 4x4 transformation matrix
                        transform_matrix = np.eye(4)
                        transform_matrix[:3, :3] = rotation_matrix
                        processed_obj["rotation_matrix"] = transform_matrix
                    elif rotation_matrix.shape == (4, 4):
                        # 4x4 transformation matrix
                        processed_obj["rotation_matrix"] = rotation_matrix
                    else:
                        print(f"Warning: Object {obj_id} has incorrect rotation matrix shape: {rotation_matrix.shape}")
                elif isinstance(rotation_matrix, np.ndarray):
                    processed_obj["rotation_matrix"] = rotation_matrix
            
            processed_data[obj_id] = processed_obj
    
    return processed_data


def load_mesh_files_from_json(json_path):
    """
    Load mesh file path list from JSON file
    
    Args:
        json_path: Path to JSON file containing file path list
        
    Returns:
        List[Path]: List of mesh file paths
    """
    with open(json_path, 'r') as f:
        file_paths = json.load(f)
    
    # Convert string paths to Path objects
    mesh_files = [Path(file_path) for file_path in file_paths]
    return mesh_files


def find_mesh_files(objaverse_dir, file_extension):
    """
    Scan all mesh files of specified format in Objaverse dataset
    """
    mesh_files = []
    objaverse_path = Path(objaverse_dir)
    
    # Ensure extension format is correct
    if not file_extension.startswith('.'):
        file_extension = '.' + file_extension
    
    # Recursively search for all files with specified format
    pattern = f"*{file_extension}"
    for mesh_file in objaverse_path.rglob(pattern):
        mesh_files.append(mesh_file)
    
    return mesh_files


def process_objaverse_dataset(mesh_files_source, canoverse_data_path, output_dir, num_processes=None):
    """
    Main function for processing Objaverse dataset
    Supports loading glb or obj format mesh files, uniformly saves as obj format
    
    Args:
        mesh_files_source: Mesh file source, can be:
                          - JSON file path (containing file path list)
                          - Directory path + file extension tuple: (dir_path, file_extension)
        canoverse_data_path: Path to canoverse data JSON file (containing object ID, category and rotation matrix info)
        output_dir: Output directory path
        num_processes: Number of processes, default None uses all CPU cores
    """
    # Load canoverse data
    print("Loading canoverse data...")
    canoverse_data = load_canoverse_data(canoverse_data_path)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load mesh file list based on input type
    if isinstance(mesh_files_source, (str, Path)):
        # Load file path list from JSON file
        print(f"Loading mesh file paths from JSON file: {mesh_files_source}")
        mesh_files = load_mesh_files_from_json(mesh_files_source)
        print(f"Loaded {len(mesh_files)} file paths from JSON file")
    elif isinstance(mesh_files_source, tuple) and len(mesh_files_source) == 2:
        # Scan files of specified format from directory
        objaverse_dir, file_extension = mesh_files_source
        supported_formats = ['obj', 'glb']
        if file_extension.lower() not in supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}, please use one of {supported_formats}")
        
        print(f"Scanning {file_extension} files in directory: {objaverse_dir}")
        mesh_files = find_mesh_files(objaverse_dir, file_extension)
        print(f"Found {len(mesh_files)} {file_extension} files")
    else:
        raise ValueError("mesh_files_source must be either a JSON file path or (directory_path, file_extension) tuple")
    
    # Filter files that exist in canoverse_data
    valid_files = []
    for mesh_file in mesh_files:
        obj_id = mesh_file.stem
        if obj_id in canoverse_data:
            valid_files.append((mesh_file, obj_id, canoverse_data[obj_id]))
    
    print(f"After filtering, {len(valid_files)} files remain")
    
    # Set number of processes
    if num_processes is None:
        num_processes = min(32, len(valid_files))
    
    print(f"Using {num_processes} processes to handle files")
    
    # Prepare multiprocessing arguments
    process_args = [(mesh_file, obj_id, obj_data, output_path) 
                   for mesh_file, obj_id, obj_data in valid_files]
    
    # Process files using multiprocessing
    processed_count = 0
    failed_count = 0
    failed_files = []
    
    if num_processes == 1:
        # Single process mode
        for args in tqdm(process_args, desc="Processing files"):
            success, obj_id, error_msg = process_single_file(args)
            if success:
                processed_count += 1
            else:
                failed_count += 1
                failed_files.append((obj_id, error_msg))
    else:
        # Multiprocessing mode
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap(process_single_file, process_args), 
                              total=len(process_args), desc="Processing files"))
            
            for success, obj_id, error_msg in results:
                if success:
                    processed_count += 1
                else:
                    failed_count += 1
                    failed_files.append((obj_id, error_msg))
    
    # Print failed file information
    if failed_files:
        print(f"\nFailed file details:")
        for obj_id, error_msg in failed_files[:10]:  # Only show first 10 failed files
            print(f"  {obj_id}: {error_msg}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more failed files")
    
    print(f"Processing complete! Successfully processed {processed_count} files, failed {failed_count} files")


def process_single_file(args):
    """
    Function to process single file, used for multiprocessing calls
    
    Args:
        args: Tuple containing (mesh_file, obj_id, obj_data, output_path)
    
    Returns:
        tuple: (success: bool, obj_id: str, error_msg: str or None)
    """
    mesh_file, obj_id, obj_data, output_path = args
    
    try:
        # Load mesh
        mesh = trimesh.load(str(mesh_file), force='mesh', process=True, maintain_order=True)
        
        # Normalize to unit sphere
        mesh = normalize_mesh_to_unit_sphere(mesh)
        
        # Check if rotation matrix exists and apply it
        if "rotation_matrix" in obj_data and obj_data["rotation_matrix"] is not None:
            try:
                rotation_matrix = obj_data["rotation_matrix"]
                # Apply rotation transformation
                mesh.apply_transform(rotation_matrix)
            except Exception as e:
                return (False, obj_id, f"Error applying rotation matrix: {e}")
        else:
            return (False, obj_id, "No rotation matrix provided")
        # Get category information
        category = obj_data.get("category", "unknown")
        
        # Create output directory structure: output_dir/category/obj_id/
        category_dir = output_path / category
        obj_output_dir = category_dir / obj_id
        obj_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save processed file
        output_file = obj_output_dir / f"{obj_id}.obj"
        mesh.export(str(output_file))
        
        return (True, obj_id, None)
        
    except Exception as e:
        return (False, obj_id, str(e))


def main():
    parser = argparse.ArgumentParser(description='Process Objaverse dataset')
    
    # Create mutually exclusive parameter group: either use JSON file or directory scanning
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--mesh_files_json', help='JSON file containing list of mesh file paths')
    input_group.add_argument('--objaverse_dir', help='Objaverse dataset directory path')
    
    parser.add_argument('--canoverse_data', required=True, help='Canoverse data JSON file path (containing object ID, category and rotation matrix info)')
    parser.add_argument('--output_dir', required=True, help='Output directory path')
    parser.add_argument('--file_extension', default='obj', choices=['obj', 'glb'], 
                       help='When using --objaverse_dir, specify mesh file extension format (obj or glb)')
    parser.add_argument('--num_processes', type=int, default=None,
                       help='Number of processes, default None uses up to 32 CPU cores')
    
    args = parser.parse_args()
    # Determine mesh file source based on input parameters
    if args.mesh_files_json:
        mesh_files_source = args.mesh_files_json
    else:
        mesh_files_source = (args.objaverse_dir, args.file_extension)
    
    process_objaverse_dataset(
        mesh_files_source,
        args.canoverse_data,
        args.output_dir,
        args.num_processes
    )


if __name__ == "__main__":
    main()
