import os
import cv2
import imageio
import shutil
import time

def create_directory(directory):
# 디렉토리 생성 함수
    if not os.path.exists(directory):
        os.makedirs(directory)
def calculate_similarity(hist1, hist2,methods):
# 이미지 간의 유사도 비교 함수
    return cv2.compareHist(hist1, hist2, methods)

methods_list = {'CORREL': cv2.HISTCMP_CORREL, # 색조와 채도가 비슷한 이미지 높을수록 비슷
               'INTERSECT': cv2.HISTCMP_INTERSECT,  # 교차점 비율 (두 이미지가 부분적으로 겹치는 경우) 높을수록 비슷
               'CHISQR':cv2.HISTCMP_CHISQR,  # 카이제곱 통계량 낮을수록 비슷 (히스토그램 분포의 차이를 나타내므로, 두 이미지의 히스토그램이 서로 유사한 경우)
               'BHATTACHARYYA':cv2.HISTCMP_BHATTACHARYYA} # 바타차르야 거리 (비슷한 특성) 낮을 수록 비슷

# 나중에 추가
methods = ''
# methods = methods_list['BHATTACHARYYA']

bf_path = os.getcwd()+'/img/bf_img/' # 현재 디렉토리 내 모든 PNG 파일 경로 리스트
temp_path = os.getcwd()+"/img/temp/" # temp 폴더 생성
create_directory(temp_path)          # 디렉토리 생성

# 이미지 temp folder copy
for file_name in os.listdir(bf_path):
    file_path = os.path.join(bf_path, file_name)
    if os.path.isfile(file_path):
        shutil.copy(file_path, os.path.join(temp_path, file_name))

path = temp_path
png_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.png')]

imgs = {}
hists = []
# 이미지 로드 및 히스토그램 계산
for file_path in png_files:
    img = cv2.imread(file_path)
    imgs[file_path] = img
    # HSV 색변환
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # 색상 정보가 필요할 경우
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 색상 정보가 필요없을 경우
    # calcHist (히스토그램 계산 함수)사용 / 이미지 색상(hue),채도(saturation) 2차원 히스토그램 계산
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256]) # 0번(h)은 0 ~ 180 사이, 1번(s)은 0 ~ 255 사이
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX) # 정규화 OpenCV의 normalize 함수 사용
    hists.append(hist)

similarity_threshold = 0.90
image_groups = []
# 유사한 이미지 그룹 찾기
for i, hist1 in enumerate(hists):
    similar_images = [i]  # 자기 자신을 포함한 그룹
    for j, hist2 in enumerate(hists):
        if methods:
            if i != j and calculate_similarity(hist1, hist2,methods_list['CORREL']) >= similarity_threshold \
                      and calculate_similarity(hist1, hist2,methods) >= similarity_threshold:
                similar_images.append(j)
        else:
            if i != j and calculate_similarity(hist1, hist2,methods_list['CORREL']) >= similarity_threshold:
                similar_images.append(j)
    image_groups.append(similar_images)


unique_image_groups = []
# 중복 제거한 유일한 그룹 찾기
for group in image_groups:
    unique_group = list(set(group))
    if unique_group not in unique_image_groups:
        unique_image_groups.append(unique_group)
# 비슷한 이미지 idx 리스트
print(unique_image_groups)

# 그룹별로 이미지 크기를 동일하게 맞추고 GIF 생성
for idx, same_imgs in enumerate(unique_image_groups):
    if len(same_imgs) > 1:
        # Group 폴더 생성
        group_directory = os.path.join(path, '../af_img/same')
        create_directory(group_directory)
        # print(same_imgs)
        # 해당 그룹에 속하는 이미지들을 same 폴더로 이동
        for img_idx in same_imgs:
            # time.sleep(2)
            img_path = png_files[img_idx]
            img_name = os.path.basename(img_path)
            target_path = os.path.join(group_directory, img_name)
            # print('img_path : ',img_path)
            # print('target_path : ', target_path)
            shutil.move(img_path, target_path)

        output_gif_path = os.path.join(path, '../af_img/gif', f'similar_images_group_{idx}.gif')
        create_directory(os.path.dirname(output_gif_path))
        images_for_gif = []

        # 해당 그룹에 속하는 이미지들의 크기 중 최대 크기를 구함
        max_width, max_height = 0, 0
        for img_idx in same_imgs:
            img = imgs[png_files[img_idx]]
            max_width = max(max_width, img.shape[1])
            max_height = max(max_height, img.shape[0])

        # 모든 이미지를 최대 크기로 조정하여 images_for_gif에 추가
        for img_idx in same_imgs:
            img = imgs[png_files[img_idx]]
            resized_img = cv2.resize(img, (max_width, max_height))
            rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            images_for_gif.append(rgb_img)

        # GIF 파일 생성
        imageio.mimsave(output_gif_path, images_for_gif, duration=0.2)
        print(f"Similar images GIF {idx+1} created successfully.")

# 이미지 파일을 img 폴더로 이동
img_directory = os.path.join(path, '../af_img/img')
create_directory(img_directory)

for file in os.listdir(path):
    if file.endswith('.png') and os.path.join(path, file) not in [os.path.join(path, '../af_img/same', img) for img in os.listdir(os.path.join(path, '../af_img/same'))]:
        img_path = os.path.join(path, file)
        target_path = os.path.join(img_directory, file)
        shutil.move(img_path, target_path)

shutil.rmtree(path)

print("Files moved successfully.")

