import fitz  # PyMuPDF
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import io
import os
import concurrent.futures

def calculate_brightness(image):
    """计算图像的整体亮度。"""
    img = Image.open(io.BytesIO(image))
    grayscale_img = img.convert("L")  # 转为灰度图像
    histogram = grayscale_img.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)

    for i in range(scale):
        ratio = histogram[i] / pixels
        brightness += ratio * (i - scale)

    return brightness / scale

def process_image_pil(image):
    """处理图像。"""
    pil_image = Image.open(io.BytesIO(image))

    # 调整对比度
    enhancer = ImageEnhance.Contrast(pil_image)
    contrast_image = enhancer.enhance(2)  # 对比度因子，可以调整

    # 锐化图像
    sharp_image = contrast_image.filter(ImageFilter.SHARPEN)

    # 复制背景图层
    layer = sharp_image.copy()

    # 高斯模糊
    blurred_layer = layer.filter(ImageFilter.GaussianBlur(100))

    # 将图层模式改为划分
    sharp_np = np.array(sharp_image, dtype=np.float32)
    blurred_np = np.array(blurred_layer, dtype=np.float32) + 1e-5  # 防止除0错误
    divided_np = np.clip(sharp_np / blurred_np * 255.0, 0, 255).astype(np.uint8)
    divided_image = Image.fromarray(divided_np)

    # 转回字节数据
    img_byte_arr = io.BytesIO()
    divided_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    return img_byte_arr

def process_image_on_page(page_num, xref, image_bytes, img_rect, brightness_threshold):
    try:
        # 计算图像的整体亮度
        brightness = calculate_brightness(image_bytes)

        # 只处理亮度低于阈值的图像
        if brightness < brightness_threshold:
            # 使用PIL处理图像
            processed_image = process_image_pil(image_bytes)

            # 确保处理后的图像尺寸与原始图像矩形区域尺寸匹配
            processed_image_pil = Image.open(io.BytesIO(processed_image))
            processed_image_pil = processed_image_pil.resize((int(img_rect.width), int(img_rect.height)), Image.LANCZOS)

            # 再次检查亮度，确保不插入过暗的图像
            final_brightness = calculate_brightness(processed_image)
            if final_brightness < brightness_threshold:
                print(f"Processed image on page {page_num}, xref {xref} is too dark after processing, skipping.")
                return None

            # 转回字节数据
            img_byte_arr = io.BytesIO()
            processed_image_pil.save(img_byte_arr, format='JPEG')
            processed_image = img_byte_arr.getvalue()

            return (page_num, img_rect, processed_image)
        else:
            print(f"Brightness too high for page {page_num}, xref {xref}, skipping this image.")
            return None  # Skip processing and return None if brightness is above threshold
    except Exception as e:
        print(f"Error processing page {page_num}, image xref {xref}: {e}")
        return None

def process_pdf(input_pdf_path, output_pdf_path, brightness_threshold=100):
    """处理PDF中的较暗图像。"""
    # 打开PDF文件
    pdf_document = fitz.open(input_pdf_path)
    tasks = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            image_list = page.get_images(full=True)

            for img in image_list:
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]

                # 获取图像的位置和大小
                img_rects = page.get_image_rects(xref)
                if not img_rects:
                    print(f"No image rect found for page {page_num}, image xref {xref}")
                    continue
                img_rect = img_rects[0]

                # 提交图像处理任务
                tasks.append(executor.submit(process_image_on_page, page_num, xref, image_bytes, img_rect, brightness_threshold))

        for task in concurrent.futures.as_completed(tasks):
            result = task.result()
            if result is not None:
                page_num, img_rect, processed_image = result
                page = pdf_document.load_page(page_num)
                page.clean_contents()
                page.insert_image(img_rect, stream=processed_image, keep_proportion=True)

    # 保存新的PDF文件
    pdf_document.save(output_pdf_path)

if __name__ == '__main__':
    # 输入和输出PDF文件路径
    input_pdf_path = r"D:\TEST.pdf"
    output_pdf_path = os.path.splitext(input_pdf_path)[0] + "改1.pdf"

    process_pdf(input_pdf_path, output_pdf_path)
