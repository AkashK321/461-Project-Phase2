import os
import re
import signal

from aws_modules.api_utils import make_response
from aws_modules.registry.system import dynamodb, TABLE_NAME, logger
from aws_modules.registry.parsing import version_satisfies, _timeout_handler

DEFAULT_PAGE_SIZE = int(os.getenv("DEFAULT_PAGE_SIZE", "10"))


def get_all_items_from_db(table):
    """
    Robustly scans the entire table handling pagination via LastEvaluatedKey.
    """
    items = []
    done = False
    start_key = None
    while not done:
        scan_kwargs = {}
        if start_key:
            scan_kwargs["ExclusiveStartKey"] = start_key

        response = table.scan(**scan_kwargs)
        items.extend(response.get("Items", []))
        start_key = response.get("LastEvaluatedKey")
        if not start_key:
            done = True
    return items


def search_artifacts(query_array, query_params):
    """
    Search artifacts with pagination based on OFFSET (record index), not page number.
    """
    # Parse offset as an integer record index (default 0)
    try:
        offset_val = int(query_params.get("offset", "0"))
    except Exception:
        offset_val = 0

    if offset_val < 0:
        offset_val = 0

    page_size = DEFAULT_PAGE_SIZE

    # Fetch ALL items to handle filtering/sorting correctly in memory
    tbl = dynamodb.Table(TABLE_NAME)
    all_items = get_all_items_from_db(tbl)

    # Filter
    if not query_array or not isinstance(query_array, list) or len(query_array) == 0:
        matched_items = all_items
    else:
        matched_items = []
        seen_ids = set()

        for query in query_array:
            q_name = (query.get("name") or "").strip()
            q_types = set(str(t).lower() for t in query.get("types", []))
            q_version = query.get("version") or query.get("version_range")

            if q_name == "*":
                q_name = ""

            for item in all_items:
                if item["id"] in seen_ids:
                    continue

                # 1. Name
                if q_name and item.get("model_name") != q_name:
                    continue

                # 2. Type
                if q_types and str(item.get("type", "")).lower() not in q_types:
                    continue

                # 3. Version
                if q_version and not version_satisfies(item.get("version"), q_version):
                    continue

                matched_items.append(item)
                seen_ids.add(item["id"])

    # Sort
    def _sort_key(it):
        return (str(it.get("model_name", "")), str(it.get("version", "")))

    matched_items.sort(key=_sort_key)

    # Paginate based on offset index
    start = offset_val
    end = start + page_size
    page_items = matched_items[start:end]

    results = [
        {"name": it.get("model_name"), "id": it.get("id"), "type": it.get("type")}
        for it in page_items
    ]

    headers = {}
    # If there are more items after this batch, return the next offset
    if end < len(matched_items):
        headers = {"Offset": str(end)}

    return make_response(200, results, headers)


def search_by_regex(body):
    """
    Searches for artifacts using a regular expression over names and README content.
    Includes ReDoS protection using signal.setitimer.
    """
    regex = body.get("regex")
    logger.info(f"Entering search_by_regex with regex='{regex}'")

    if not regex:
        logger.warning("search_by_regex: Missing regex in body")
        return make_response(400, {"error": "Missing regex"})

    try:
        pattern = re.compile(regex)
    except re.error as e:
        logger.warning(f"search_by_regex: Invalid regex '{regex}': {e}")
        return make_response(400, {"error": "Invalid regex"})

    tbl = dynamodb.Table(TABLE_NAME)

    # Scan all items using helper to ensure full table coverage
    items = get_all_items_from_db(tbl)

    logger.info(f"search_by_regex: Scanned {len(items)} items")

    # Set up signal handler for ReDoS protection
    signal.signal(signal.SIGALRM, _timeout_handler)

    matches = []

    try:
        for item in items:
            name = item.get("model_name", "")
            readme = item.get("readme", "")

            is_match = False

            # 1. Check Name
            signal.setitimer(signal.ITIMER_REAL, 0.1)  # 100ms
            try:
                if pattern.search(name):
                    is_match = True
            except TimeoutError:
                raise TimeoutError("Regex execution timed out on name")
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)

            if is_match:
                logger.info(
                    f"Match found for regex '{regex}' \
                    on item: {name} (ID: {item.get('id')})"
                )
                matches.append(
                    {"name": name, "id": item.get("id"), "type": item.get("type")}
                )
                continue  # Skip checking readme if name matched

            # 2. Check Readme
            signal.setitimer(signal.ITIMER_REAL, 0.1)  # 100ms
            try:
                if pattern.search(readme):
                    is_match = True
            except TimeoutError:
                raise TimeoutError("Regex execution timed out on readme")
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)

            if is_match:
                matches.append(
                    {"name": name, "id": item.get("id"), "type": item.get("type")}
                )

    except TimeoutError:
        logger.warning(f"ReDoS detected for regex: {regex}")
        return make_response(400, {"error": "Regex too complex (ReDoS detected)"})
    except Exception as e:
        logger.error(f"Error during regex search: {e}")
        return make_response(500, {"error": f"Internal error during search: {e}"})

    logger.info(f"search_by_regex: Found {len(matches)} matches")

    if not matches:
        logger.info("search_by_regex: No matches found, returning 404")
        return make_response(404, {"error": "No artifact found under this regex."})

    response = make_response(200, matches)
    logger.info(f"search_by_regex returning success with {len(matches)} items")
    return response
