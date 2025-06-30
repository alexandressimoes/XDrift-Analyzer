import numpy as np
import pandas as pd
from typing import Dict, List, Union, Any, Optional

class DriftCharacterizer:
    """Characterizes how features have drifted between reference and current data."""
    
    def characterize(self, drift_results: Dict[str, Dict[str, Any]], 
                    reference: pd.DataFrame, current: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Characterize drift for features that have drifted.
        
        Args:
            drift_results: Results from DriftDetector
            reference: Reference data
            current: Current data
            
        Returns:
            Dictionary with detailed drift characterization for each feature
        """
        characterization = {}
        
        for feature, result in drift_results.items():
            if not result["drift_detected"]:
                continue
                
            if result["feature_type"] == "numerical":
                characterization[feature] = self._characterize_numerical(
                    reference[feature], current[feature]
                )
            else:
                characterization[feature] = self._characterize_categorical(
                    reference[feature], current[feature]
                )
                
        return characterization
    
    def _characterize_numerical(self, reference: pd.Series, current: pd.Series) -> Dict[str, Any]:
        """Characterize drift in numerical features."""
        # Calculate statistics
        ref_stats = {
            "mean": reference.mean(),
            "median": reference.median(),
            "std": reference.std(),
            "min": reference.min(),
            "max": reference.max(),
            "q25": reference.quantile(0.25),
            "q75": reference.quantile(0.75)
        }
        
        curr_stats = {
            "mean": current.mean(),
            "median": current.median(),
            "std": current.std(),
            "min": current.min(),
            "max": current.max(),
            "q25": current.quantile(0.25),
            "q75": current.quantile(0.75)
        }
        
        # Calculate differences
        diff = {}
        percent_diff = {}
        for stat in ref_stats:
            diff[stat] = curr_stats[stat] - ref_stats[stat]
            percent_diff[stat] = (diff[stat] / ref_stats[stat] * 100 
                                if ref_stats[stat] != 0 else float("inf"))
        
        # Find major changes
        major_changes = []
        
        # Check for shifts in central tendency
        if abs(percent_diff["mean"]) > 10:
            major_changes.append("mean_shift")
        if abs(percent_diff["median"]) > 10:
            major_changes.append("median_shift")
            
        # Check for changes in spread
        if abs(percent_diff["std"]) > 20:
            major_changes.append("spread_change")
            
        # Check for range changes
        if abs(percent_diff["min"]) > 20 or abs(percent_diff["max"]) > 20:
            major_changes.append("range_change")
            
        # Check for skewness changes
        ref_skew = reference.skew()
        curr_skew = current.skew()
        if abs(ref_skew - curr_skew) > 0.5:
            major_changes.append("skewness_change")
        
        return {
            "reference_stats": ref_stats,
            "current_stats": curr_stats,
            "absolute_diff": diff,
            "percent_diff": percent_diff,
            "major_changes": major_changes,
            "summary": self._generate_numerical_summary(ref_stats, curr_stats, percent_diff, major_changes)
        }
    
    def _characterize_categorical(self, reference: pd.Series, current: pd.Series) -> Dict[str, Any]:
        """Characterize drift in categorical features."""
        # Calculate frequencies
        ref_freqs = reference.value_counts(normalize=True).to_dict()
        curr_freqs = current.value_counts(normalize=True).to_dict()
        
        # Find all categories
        all_categories = list(set(ref_freqs.keys()) | set(curr_freqs.keys()))
        
        # Calculate differences
        changes = {}
        for category in all_categories:
            ref_freq = ref_freqs.get(category, 0)
            curr_freq = curr_freqs.get(category, 0)
            
            abs_diff = curr_freq - ref_freq
            # Avoid division by zero
            rel_diff = abs_diff / ref_freq * 100 if ref_freq > 0 else float("inf")
            
            changes[category] = {
                "reference_freq": ref_freq,
                "current_freq": curr_freq,
                "absolute_diff": abs_diff,
                "relative_diff": rel_diff
            }
        
        # Categorize major changes
        new_categories = [cat for cat in all_categories if cat in curr_freqs and cat not in ref_freqs]
        missing_categories = [cat for cat in all_categories if cat in ref_freqs and cat not in curr_freqs]
        
        # Find categories with significant changes
        significant_changes = {
            cat: changes[cat] for cat in changes 
            if cat not in new_categories and cat not in missing_categories and 
            abs(changes[cat]["absolute_diff"]) > 0.05  # More than 5% change
        }
        
        return {
            "category_changes": changes,
            "new_categories": new_categories,
            "missing_categories": missing_categories,
            "significant_changes": significant_changes,
            "summary": self._generate_categorical_summary(
                new_categories, missing_categories, significant_changes)
        }
    
    def _generate_numerical_summary(self, ref_stats: Dict[str, float], 
                                  curr_stats: Dict[str, float],
                                  percent_diff: Dict[str, float],
                                  major_changes: List[str]) -> str:
        """Generate a human-readable summary of numerical drift."""
        summary_parts = []
        
        if "mean_shift" in major_changes:
            direction = "increased" if percent_diff["mean"] > 0 else "decreased"
            summary_parts.append(
                f"The mean has {direction} by {abs(percent_diff['mean']):.1f}% "
                f"(from {ref_stats['mean']:.2f} to {curr_stats['mean']:.2f})."
            )
            
        if "spread_change" in major_changes:
            direction = "increased" if percent_diff["std"] > 0 else "decreased"
            summary_parts.append(
                f"The standard deviation has {direction} by {abs(percent_diff['std']):.1f}% "
                f"(from {ref_stats['std']:.2f} to {curr_stats['std']:.2f})."
            )
            
        if "range_change" in major_changes:
            summary_parts.append(
                f"The range has changed significantly: min from {ref_stats['min']:.2f} to "
                f"{curr_stats['min']:.2f}, max from {ref_stats['max']:.2f} to {curr_stats['max']:.2f}."
            )
            
        if not summary_parts:
            summary_parts.append("Minor statistical changes observed across the distribution.")
            
        return " ".join(summary_parts)
    
    def _generate_categorical_summary(self, new_categories: List[str], 
                                    missing_categories: List[str],
                                    significant_changes: Dict[str, Dict[str, float]]) -> str:
        """Generate a human-readable summary of categorical drift."""
        summary_parts = []
        
        if new_categories:
            if len(new_categories) <= 3:
                summary_parts.append(f"New categories appeared: {', '.join(new_categories)}.")
            else:
                summary_parts.append(f"{len(new_categories)} new categories appeared.")
                
        if missing_categories:
            if len(missing_categories) <= 3:
                summary_parts.append(f"Categories disappeared: {', '.join(missing_categories)}.")
            else:
                summary_parts.append(f"{len(missing_categories)} categories disappeared.")
                
        if significant_changes:
            top_changes = sorted(
                significant_changes.items(), 
                key=lambda x: abs(x[1]["absolute_diff"]),
                reverse=True
            )[:3]
            
            for cat, change in top_changes:
                direction = "increased" if change["absolute_diff"] > 0 else "decreased"
                summary_parts.append(
                    f"Category '{cat}' {direction} from {change['reference_freq']:.1%} to {change['current_freq']:.1%}."
                )
                
        if not summary_parts:
            summary_parts.append("Minor changes in category frequencies observed.")
            
        return " ".join(summary_parts)
