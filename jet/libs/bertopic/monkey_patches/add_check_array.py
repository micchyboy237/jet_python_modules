# monkey_patches.py
# --- BERTopic + HDBSCAN compatibility fix for scikit-learn 1.8+ ---

import sys

import sklearn.utils


def init_patch(force=False):
    """
    Initialize the monkey patch for sklearn.utils.check_array.
    Call this as early as possible (before importing bertopic/hdbscan).

    Args:
        force (bool): Force re-apply even if already patched.
    """
    patch_name = "bertopic_hdbscan_sklearn_patch"

    # Check if already patched (unless force=True)
    if not force and patch_name in getattr(sys, "_applied_patches", {}):
        print("ℹ️  Monkey patch already applied.")
        return True

    try:
        # Store original function
        original_check_array = sklearn.utils.check_array

        def patched_check_array(*args, **kwargs):
            """Patched version that handles the renamed parameter."""
            if "force_all_finite" in kwargs:
                kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")

            # Debug log on first use
            if not hasattr(patched_check_array, "_logged"):
                patched_check_array._logged = True
                print(
                    "✅ [Monkey Patch] check_array: force_all_finite → ensure_all_finite applied"
                )

            return original_check_array(*args, **kwargs)

        # Apply the patch
        sklearn.utils.check_array = patched_check_array

        # Track that we patched it
        if not hasattr(sys, "_applied_patches"):
            sys._applied_patches = {}
        sys._applied_patches[patch_name] = True

        print("✅ BERTopic/HDBSCAN monkey patch successfully applied!")
        return True

    except Exception as e:
        print(f"❌ Failed to apply monkey patch: {e}")
        return False


# Optional: Auto-apply when the module is imported
if __name__ != "__main__":
    init_patch()
