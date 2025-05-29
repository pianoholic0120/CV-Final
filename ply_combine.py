import pymeshlab as ml
import argparse

def main():
    parser = argparse.ArgumentParser(description="Combine two PLY files into one.")
    parser.add_argument("--input1", help="First input PLY file to combine.")
    parser.add_argument("--input2", help="Second input PLY file to combine.")
    parser.add_argument("--output", help="Output PLY file.")
    args = parser.parse_args()
    
    # Create MeshSet and load meshes
    mesh = ml.MeshSet()
    mesh.load_new_mesh(args.input1)
    mesh.load_new_mesh(args.input2)
    
    mesh.apply_filter('meshing_merge_close_vertices')  
    mesh.generate_by_merging_visible_meshes()
    
    # Save the combined mesh
    mesh.save_current_mesh(args.output)
    print(f"Successfully combined meshes and saved to: {args.output}")

if __name__ == "__main__":
    main()