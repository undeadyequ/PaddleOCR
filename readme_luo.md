# inference
    - For 
python3 tools/infer/predict_system.py --image_dir="test_image/picture1.png" --det_model_dir="server/ch_ppocr_server_v2.0_det_infer" --rec_model_dir="./server/ch_ppocr_server_v2.0_rec_infer" --cls_model="server/ch_ppocr_mobile_v2.0_cls_infer"  --use_angle_cls=True --use_gpu=False
    
    - For debug path
python3 tools/infer/predict_system.py --image_dir="../../test_image/picture1.png" --det_model_dir="../../server/ch_ppocr_server_v2.0_det_infer" --rec_model_dir="../../server/ch_ppocr_server_v2.0_rec_infer" --cls_model="../../server/ch_ppocr_mobile_v2.0_cls_infer"  --use_angle_cls=True --use_gpu=False --rec_char_dict_path "../../ppocr/utils/ppocr_keys_v1.txt" --vis_font_path "../../doc/simfang.ttf"


# train
python3 tools/train.py -c configs/det/det_mv3_db.yml


python train.py --data_dir /home/Data/image_data/UCLA-protest/ --batch_size 32 --lr 0.002 --print_freq 100 --epochs 100 --cuda



torch.Size([32, 3, 224, 224])

numpy(270, 272, 3)

