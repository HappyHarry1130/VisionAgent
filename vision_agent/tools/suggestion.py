from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Set, Tuple, cast

import numpy as np
from vision_agent.configs import Config
from vision_agent.lmm.lmm import LMM
from vision_agent.utils.image_utils import convert_to_b64

from vision_agent.agent.agent_utils import extract_tag
from vision_agent.agent.visual_design_patterns import (
    CHECK_COLOR,
    COMPARING_SIZES,
    DEPTH_POSITION,
    FINDING_FEATURES_WITH_VIDEO_TRACKING,
    LARGE_IMAGE,
    MISSING_GRID_ELEMENTS,
    MISSING_HORIZONTAL_ELEMENTS,
    MISSING_VERTICAL_ELEMENTS,
    NESTED_STRUCTURE,
    RELATIVE_POSITION,
    SMALL_TEXT,
    SUGGESTIONS,
)

CONFIG = Config()


def run_multi_judge(
    suggester: LMM,
    user_request: str,
    image_size_info: str,
    media: List[str],
) -> Tuple[Set[str], str]:
    prompt = SUGGESTIONS.format(
        user_request=user_request, image_size_info=image_size_info
    )

    def run_judge():
        response = cast(str, suggester.generate(prompt, media=media, temperature=1.0))
        reason = extract_tag(response, "reason")
        reason = reason if reason is not None else ""
        categories = extract_tag(response, "categories")
        if categories is not None:
            elts = categories.split(",")
            elts = [
                elt.replace("[", "").replace("]", "").replace("'", "").strip()
                for elt in elts
            ]
            categories = set(elts)
        else:
            categories = set()
        return categories, reason

    responses = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_judge) for _ in range(3)]
        for future in as_completed(futures):
            responses.append(future.result())

    counts = {}
    for categories, _ in responses:
        for category in categories:
            counts[category] = counts.get(category, 0) + 1

    majority_categories = {category for category, count in counts.items() if count >= 2}
    category_to_reason = {tuple(cats): reason for cats, reason in responses}

    if tuple(majority_categories) in category_to_reason:
        return majority_categories, category_to_reason[tuple(majority_categories)]

    combined_reason = []
    for category in majority_categories:
        matching_responses = [reason for cats, reason in responses if category in cats]
        if matching_responses:
            combined_reason.append(matching_responses[0])

    return majority_categories, "\n".join(combined_reason)


def suggestion_impl(prompt: str, medias: List[np.ndarray]) -> str:
    suggester = CONFIG.create_suggester()
    if isinstance(medias, np.ndarray):
        medias = [medias]
    all_media_b64 = [
        "data:image/png;base64," + convert_to_b64(media) for media in medias
    ]
    image_sizes = [media.shape for media in medias]
    image_size_info = (
        " The original image sizes were "
        + str(image_sizes)
        + ", I have resized them to 768x768, if my resize is much smaller than the original image size I may have missed some details."
    )

    categories, reason = run_multi_judge(
        suggester, prompt, image_size_info, all_media_b64
    )

    suggestion = ""
    i = 0
    for suggestion_and_cat in [
        LARGE_IMAGE,
        SMALL_TEXT,
        CHECK_COLOR,
        COMPARING_SIZES,
        MISSING_GRID_ELEMENTS,
        MISSING_HORIZONTAL_ELEMENTS,
        MISSING_VERTICAL_ELEMENTS,
        FINDING_FEATURES_WITH_VIDEO_TRACKING,
        NESTED_STRUCTURE,
        RELATIVE_POSITION,
        DEPTH_POSITION,
    ]:
        if len(categories & suggestion_and_cat[1]) > 0:
            suggestion += (
                f"\n[suggestion {i}]\n"
                + suggestion_and_cat[0]
                + f"\n[end of suggestion {i}]"
            )
            i += 1

    response = f"[suggestions]\n{reason}\n{suggestion}\n[end of suggestions]"
    return response
