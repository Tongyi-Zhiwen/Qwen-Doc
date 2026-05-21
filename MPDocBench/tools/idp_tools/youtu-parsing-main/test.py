from youtu_hf_parser import YoutuOCRParserHF

# Initialize the parser with model configuration
parser = YoutuOCRParserHF(
    model_path='/nas-mmu/xhd/checkpoints/official_models/tencent/Youtu-Parsing',                    # Path to downloaded model weights
    enable_angle_correct=True,                # Set to False to disable angle correction
    # angle_correct_model_path='/nas-mmu/zhoubb/code/IDP_tools/youtu-parsing-main/model.pth' # If None, model will auto-download to default path; if custom path, manually download https://github.com/TencentCloudADP/youtu-parsing/releases/download/v1.0.0/model.pth to specified location
)

# Parse a document (supports images, PDFs, and more)
parser.parse_file(
    input_path='/nas-mmu/xhd/62b3a073-1a00-416d-9c26-cb8d40793855.png',     # Input document path
    output_dir='./'      # Output directory for results
)

print("✅ Document parsing completed!")
print(f"📄 Results saved to: {output_dir}")
