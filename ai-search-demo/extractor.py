import os
import logging
import fitz
from PIL import Image
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
from azure.ai.documentintelligence.models import (
    AnalyzeDocumentRequest,
    # DocumentContentFormat,
)
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure your Azure Form Recognizer endpoint and key
AZURE_FORM_RECOGNIZER_ENDPOINT = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
AZURE_FORM_RECOGNIZER_KEY = os.getenv("AZURE_FORM_RECOGNIZER_KEY")


def crop_image_from_pdf_page(pdf_path, page_number, bounding_box):
    """
    Crops a region from a given page in a PDF and returns it as an image.

    :param pdf_path: Path to the PDF file.
    :param page_number: The page number to crop from (0-indexed).
    :param bounding_box: A tuple of (x0, y0, x1, y1) coordinates for the bounding box.
    :return: A PIL Image of the cropped area.
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)

    # Cropping the page. The rect requires the coordinates in the format (x0, y0, x1, y1).
    bbx = [x * 72 for x in bounding_box]
    rect = fitz.Rect(bbx)
    pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), clip=rect)

    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    doc.close()

    return img


def extract_document_content(pdf_path, doc_name):
    logger.info("Extracting text, tables, and figures from PDF: %s", pdf_path)
    if not os.path.exists(pdf_path):
        logger.error("PDF file does not exist: %s", pdf_path)
        return None, None, None

    text_content = ""
    tables = []
    tables_markdown = ""
    figure_paths = []

    try:
        client = DocumentIntelligenceClient(
            endpoint=AZURE_FORM_RECOGNIZER_ENDPOINT,
            credential=AzureKeyCredential(AZURE_FORM_RECOGNIZER_KEY),
        )

        with open(pdf_path, "rb") as f:
            poller = client.begin_analyze_document(
                "prebuilt-layout",
                analyze_request=f,
                content_type="application/octet-stream",
                output_content_format="markdown",  # DocumentContentFormat.MARKDOWN,
            )
        result: AnalyzeResult = poller.result()

        for page in result.pages:
            if page.lines:
                for line in page.lines:
                    text_content += line.content + " "
            # Gather tables
        if result.tables:
            for table in result.tables:
                tables_markdown = ""
                table_data = [
                    ["" for _ in range(table.column_count)]
                    for _ in range(table.row_count + 1)
                ]

                for cell in table.cells:
                    if cell.row_index == 0:
                        table_data[cell.row_index][cell.column_index] = cell.content
                        table_data[cell.row_index + 1][cell.column_index] = " --- "
                    else:
                        table_data[cell.row_index + 1][cell.column_index] = cell.content
                for row in table_data:
                    tables_markdown += "| " + " | ".join(row) + " |\n"
                tables_markdown += "\n"
                logger.info("Table content: %s", tables_markdown)
                tables.append(tables_markdown)
            logger.info("Found %d tables in the document.", len(result.tables))
        # Gather figure bounding regions (placeholder for saving images)
        if result.figures:
            logger.info("Found %d figures in the document.", len(result.figures))
            for idx, figure in enumerate(result.figures):
                for region in figure.bounding_regions:
                    bounding_box = (
                        region.polygon[0],
                        region.polygon[1],
                        region.polygon[4],
                        region.polygon[5],
                    )
                    logger.info(
                        "Figure %d bounding box coordinates: %s", idx, bounding_box
                    )
                    # Crop the image using the bounding box coordinates
                    cropped_img = crop_image_from_pdf_page(
                        pdf_path, region.page_number - 1, bounding_box
                    )

                    save_path = f"./figures/figure_{doc_name}_{idx}.jpg"
                    figure_paths.append(save_path)
                    logger.info(
                        "Saving figure to %s (bounding region only).", save_path
                    )
                    cropped_img.save(save_path)

        logger.info("Extraction completed.")
        return text_content, tables, figure_paths

    except Exception as e:
        logger.error("Error during PDF extraction: %s", str(e))
        return None, None, None


def process_pdf(pdf_path):
    text, tables, images = extract_document_content(pdf_path)

    if text is None:
        logger.error("Failed to extract text from PDF.")
        return None, None, None

    return text, tables, images


if __name__ == "__main__":
    pdf_path = "uploads/sakura_bliss_onsen_japan.pdf"
    process_pdf(pdf_path)
