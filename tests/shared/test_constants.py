from src.shared.constants import PRODUCTS, TERRALAB_URL, NEWSLETTER_URL


class TestConstants:
    def test_products_have_required_keys(self):
        for pid, info in PRODUCTS.items():
            assert "display_name" in info
            assert "qsettings_prefix" in info
            assert "newsletter_param" in info

    def test_both_products_exist(self):
        assert "ai-canvas" in PRODUCTS
        assert "ai-segmentation" in PRODUCTS

    def test_urls_are_https(self):
        assert TERRALAB_URL.startswith("https://")
        assert NEWSLETTER_URL.startswith("https://")

    def test_prefixes_are_distinct(self):
        prefixes = [p["qsettings_prefix"] for p in PRODUCTS.values()]
        assert len(prefixes) == len(set(prefixes))
