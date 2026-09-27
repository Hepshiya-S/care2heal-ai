import qrcode


def generate_reorder_qr(drug_name: str, store_url: str, output_path: str = "reorder_qr.png") -> str:
    """
    Generates a QR code that links to a store page for reordering a specific medicine.
    """
    img = qrcode.make(store_url)
    img.save(output_path)
    print(f"QR code for reordering {drug_name} saved to {output_path}")
    return output_path


def confirm_and_reorder(drug_name: str) -> str:
    """
    Simulates the 'tap to confirm' action after a QR scan or a refill-due notice.
    In a real deployment, this would call the pharmacy's order API instead of printing.
    """
    return f"Order placed: {drug_name} has been reordered. You'll be notified when it ships."


if __name__ == "__main__":
    generate_reorder_qr("Metformin", "https://www.1mg.com/search/all?name=metformin")
    result = confirm_and_reorder("Metformin")
    print(result)