from visualize_video import save_video_from_npz
from compute_MSE import process_all
from pathlib import Path
import os


if __name__ == "__main__":
    
    inference_type = "_1000/"
    input_dir =         "/home/riccardo/RaMViD/outputs"+inference_type #"/media/pinas/riccardo/outputs_448train_300step/" #
    output_avi_dir =    "/home/riccardo/RaMViD/outputs"+inference_type+"avenue_avi/" #"/media/pinas/riccardo/outputs_448train_300step/avenue_avi/" #"/home/riccardo/RaMViD/outputs"+inference_type+"avenue_avi/"
    output_dir = "/home/riccardo/results_avenue"+inference_type  # oppure un'altra directory
    os.makedirs(output_avi_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    
    original_dir =      "/home/riccardo/Avenue_Dataset/testing_videos/"
    labels_dir =        "/home/riccardo/Avenue_Dataset/masks/"
    

 

    for video_number in os.listdir(input_dir):

        path=input_dir + video_number
        #os.makedirs(path+"/avi_aggregated_format", exist_ok=True)

        # Esempio di utilizzo:
        save_video_from_npz(
                generated_path=path,
                output_file=output_avi_dir + video_number +".avi",
                fps=25
            )
        

    process_all(output_avi_dir, original_dir, labels_dir, output_dir,multiple_graphs=True,roc_graph=True,)