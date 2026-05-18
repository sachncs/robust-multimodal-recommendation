import gzip
import os
import tempfile
from unittest.mock import MagicMock, patch

from rmr.data.download import download_amazon_dataset, download_file


def test_download_file():
    with tempfile.TemporaryDirectory() as tmpdir, patch(
        "rmr.data.download.urllib.request.urlopen"
    ) as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.side_effect = [b"fake data", b""]
            mock_resp.__enter__.return_value = mock_resp
            mock_urlopen.return_value = mock_resp
            path = os.path.join(tmpdir, "test.gz")
            download_file("http://example.com/fake.gz", path)
            assert os.path.exists(path)


def test_download_amazon_dataset_mocked():
    with tempfile.TemporaryDirectory() as tmpdir, patch(
        "rmr.data.download.download_file"
    ) as mock_download:
            category = "FakeCategory"
            review_file = f"{category}_5.json.gz"
            meta_file = f"{category}_metadata.json.gz"
            # Pre-create gzipped dummy files so download_file won't be called for real,
            # but since we mock it, we need to create the .gz files ourselves
            for fname in [review_file, meta_file]:
                gz_path = os.path.join(tmpdir, fname)
                with gzip.open(gz_path, "wb") as f_out:
                    f_out.write(b'{"reviewerID":"U1","asin":"I1"}\n')
            download_amazon_dataset(category, data_dir=tmpdir)
            # After download_amazon_dataset, .gz should be removed and .json should exist
            for fname in [review_file, meta_file]:
                json_path = os.path.join(tmpdir, fname.replace(".gz", ""))
                gz_path = os.path.join(tmpdir, fname)
                assert os.path.exists(json_path)
                assert not os.path.exists(gz_path)
            # download_file should have been called for each file
            assert mock_download.call_count == 2


def test_download_amazon_dataset_skips_if_exists():
    with tempfile.TemporaryDirectory() as tmpdir, patch(
        "rmr.data.download.download_file"
    ) as mock_download:
            category = "FakeCategory"
            review_file = f"{category}_5.json.gz"
            meta_file = f"{category}_metadata.json.gz"
            # Pre-create the unzipped files so download is skipped
            for fname in [review_file, meta_file]:
                json_path = os.path.join(tmpdir, fname.replace(".gz", ""))
                with open(json_path, "w") as f:
                    f.write('{"reviewerID":"U1","asin":"I1"}\n')
            download_amazon_dataset(category, data_dir=tmpdir)
            mock_download.assert_not_called()
