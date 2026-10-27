import argparse

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FilterSelector

from core.settings import get_settings


def delete_collection_data(
    client: QdrantClient,
    collection_name: str,
    *,
    confirm: bool = False,
) -> int:
    """Count points and, only when confirmed, delete all points in a collection."""
    if not client.collection_exists(collection_name):
        raise ValueError(f"Qdrant collection does not exist: {collection_name}")

    point_count = client.count(collection_name=collection_name, exact=True).count
    if not confirm:
        return point_count

    client.delete(
        collection_name=collection_name,
        points_selector=FilterSelector(filter=Filter()),
        wait=True,
    )
    remaining_count = client.count(collection_name=collection_name, exact=True).count
    if remaining_count:
        raise RuntimeError(
            f"Deletion did not clear collection {collection_name!r}; "
            f"{remaining_count} points remain"
        )
    return point_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Count or delete all points in a Qdrant collection."
    )
    parser.add_argument(
        "--collection",
        help="Collection name (defaults to QDRANT_COLLECTION or 'FinAi').",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually delete all points. Without this option, only count them.",
    )
    args = parser.parse_args()

    settings = get_settings()
    qdrant_url = settings.qdrant_url
    if not qdrant_url:
        parser.error("QDRANT_URL is missing from the environment or project .env")

    collection_name = args.collection or settings.qdrant_collection
    client = QdrantClient(
        url=qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    try:
        point_count = delete_collection_data(
            client,
            collection_name,
            confirm=args.confirm,
        )
    finally:
        client.close()

    if args.confirm:
        print(f"Deleted {point_count} points from {collection_name!r}.")
    else:
        print(
            f"Dry run: {collection_name!r} contains {point_count} points. "
            "Pass --confirm to delete them."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
