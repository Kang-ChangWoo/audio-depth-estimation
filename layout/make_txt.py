import os
from tqdm import tqdm

def main():
    src_path = "/root/storage/matterport3d_0303renew/"
    dst_path = "/root/storage/matterport3d_0421_layout/"
    pred_path = "/root/storage/implementation/shared_audio/baseline/results/unet_256_soundspaces_BS128_Lr0.0005_AdamW_exp202_n2_temap_lr5e4_lsh0.1/vis_npy/"

    train_list = ["gTV8FGcVJC9", "pLe4wQe7qrG", "q9vSo1VnCiC", "sT4fr6TAbpF", "uNb9QFRL6hY", "Z6MFQCViBuw"]
    val_list = ["HxpKQynjfin"]
    test_list = ["8WUmhLawc2A", "EDJbREhghzL"]

    with open ("./train.txt", "w") as f:
        for folder in tqdm(train_list):
            for idx in range(400):
                rgb_path = src_path + folder + "/erp_rgb/" + "erp_" + str(idx).zfill(3) + ".png"
                depth_path = pred_path + folder + "/" + str(idx).zfill(3) + "_pred.npy"
                layout_path = dst_path + folder + "/erp_" + str(idx).zfill(3) + ".npy"
                f.write(rgb_path + " " + depth_path + " " + layout_path + "\n")

                idx += 1
        f.close()


    with open ("./val.txt", "w") as f:
        for folder in tqdm(val_list):
            for idx in range(400):
                rgb_path = src_path + folder + "/erp_rgb/" + "erp_" + str(idx).zfill(3) + ".png"
                depth_path = pred_path + folder + "/" + str(idx).zfill(3) + "_pred.npy"
                layout_path = dst_path + folder + "/erp_" + str(idx).zfill(3) + ".npy"
                f.write(rgb_path + " " + depth_path + " " + layout_path + "\n")

                idx += 1
        f.close()

    
    with open ("./test.txt", "w") as f:
        for folder in tqdm(test_list):
            for idx in range(400):
                rgb_path = src_path + folder + "/erp_rgb/" + "erp_" + str(idx).zfill(3) + ".png"
                depth_path = pred_path + folder + "/" + str(idx).zfill(3) + "_pred.npy"
                layout_path = dst_path + folder + "/erp_" + str(idx).zfill(3) + ".npy"
                f.write(rgb_path + " " + depth_path + " " + layout_path + "\n")

                idx += 1
        f.close()




if __name__ == '__main__':
    main()