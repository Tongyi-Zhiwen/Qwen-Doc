import re
import cairosvg
from PIL import Image, ImageDraw, ImageFont 

def fix_svg(svg: str) -> str:
    """修复不完整的 SVG 标签"""
    # 1) 定向补：末尾 <path d="... 这种 d 属性没闭合
    if re.search(r'(<path\b[^>]*\bd="[^">]*$)', svg):
        svg += '">'
    
    # 2) 删除末尾残缺标签
    svg = re.sub(r'<[^>]*$', '', svg)
    
    # 3) 顺序扫描做栈匹配，补齐未闭合标签
    stack = []
    TAG_RE = re.compile(r'</?\s*([a-zA-Z][\w:-]*)\b[^>]*?/?>')
    for m in TAG_RE.finditer(svg):
        name = m.group(1)
        token = m.group(0)
        is_close = token.lstrip().startswith("</")
        is_self_close = token.rstrip().endswith("/>")
        
        if is_self_close:
            continue
        if not is_close:
            stack.append(name)
        else:
            if name in stack[::-1]:
                while stack and stack[-1] != name:
                    stack.pop()
                if stack and stack[-1] == name:
                    stack.pop()
    
    # 4) 补齐剩余未闭合
    while stack:
        svg += f'</{stack.pop()}>'
    
    return svg


def extract_svg_from_response(response: str):
    """从模型响应中提取 SVG 内容，返回 (svg_content, success)"""
    response = response.replace("svg:", "").strip()
    
    # 尝试匹配完整的 <svg>...</svg>
    svg_match = re.search(r'<svg[^>]*>(.*?)</svg>', response, re.DOTALL)
    if svg_match:
        return svg_match.group(0), True
    
    # 尝试匹配不完整的 SVG
    svg_match = re.search(r'<svg[^>]*>.*', response, re.DOTALL)
    if svg_match:
        return fix_svg(svg_match.group(0)), True
    
    return None, False


def svg_to_png(svg_content: str, output_path: str, width: int = 1024, height: int = 1024):
    """将 SVG 转换为 PNG 图片"""
    import cairosvg
    try:
        cairosvg.svg2png(
            bytestring=svg_content.encode('utf-8'),
            write_to=output_path,
            output_width=width,
            output_height=height,
            background_color='white'
        )
        return True, None
    except Exception as e:
        return False, str(e)

def _add_label(image: Image.Image, label: str, font_size: int = 24) -> Image.Image:
    """在图片右上角添加标签"""
    draw = ImageDraw.Draw(image)
    
    # 加载字体（优先使用粗体）
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()  # 找不到就用默认字体
    
    # 计算文字位置（右上角）
    padding = 10
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = image.width - text_width - padding * 2  # 靠右
    y = padding  # 靠上
    
    # 绘制半透明黑色背景
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
        fill=(0, 0, 0, 180)  # 黑色，180 表示透明度
    )
    
    # 合并背景层
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    image = Image.alpha_composite(image, overlay)
    
    # 绘制白色文字
    draw = ImageDraw.Draw(image)
    draw.text((x, y), label, font=font, fill=(255, 255, 255, 255))
    
    return image


def create_comparison_image(original_image, rendered_image, gap=10, 
                            top_label: str = "Origin", 
                            bottom_label: str = "Generated"):
    """
    创建对比图：上面原图，下面渲染图，并添加标签
    
    Args:
        original_image: PIL Image，原图
        rendered_image: PIL Image 或路径，渲染后的图片
        gap: 两张图之间的间隔高度（像素）
        top_label: 上图标签，默认 "Origin"
        bottom_label: 下图标签，默认 "Generated"
    
    Returns:
        PIL Image: 拼接后的对比图
    """
    if isinstance(rendered_image, str):
        rendered_image = Image.open(rendered_image)
    
    # 统一宽度，按比例缩放
    target_width = max(original_image.width, rendered_image.width)
    
    # 缩放原图
    if original_image.width != target_width:
        scale = target_width / original_image.width
        new_height = int(original_image.height * scale)
        original_image = original_image.resize((target_width, new_height), Image.LANCZOS)
    
    # 缩放渲染图
    if rendered_image.width != target_width:
        scale = target_width / rendered_image.width
        new_height = int(rendered_image.height * scale)
        rendered_image = rendered_image.resize((target_width, new_height), Image.LANCZOS)
    
    # ===== 新增：转换为 RGBA 模式以便添加标签 =====
    if original_image.mode != 'RGBA':
        original_image = original_image.convert('RGBA')
    if rendered_image.mode != 'RGBA':
        rendered_image = rendered_image.convert('RGBA')
    
    # ===== 新增：添加标签 =====
    original_image = _add_label(original_image, top_label)
    rendered_image = _add_label(rendered_image, bottom_label)
    
    # ===== 修改：转回 RGB 模式进行拼接 =====
    if original_image.mode == 'RGBA':
        bg = Image.new('RGB', original_image.size, (255, 255, 255))
        bg.paste(original_image, mask=original_image.split()[3])
        original_image = bg
    
    if rendered_image.mode == 'RGBA':
        bg = Image.new('RGB', rendered_image.size, (255, 255, 255))
        bg.paste(rendered_image, mask=rendered_image.split()[3])
        rendered_image = bg
    
    # 计算拼接后的尺寸
    total_height = original_image.height + gap + rendered_image.height
    
    # 创建空白画布
    comparison = Image.new('RGB', (target_width, total_height), (255, 255, 255))
    
    # 粘贴两张图：上面原图，下面渲染图
    comparison.paste(original_image, (0, 0))
    comparison.paste(rendered_image, (0, original_image.height + gap))
    
    return comparison