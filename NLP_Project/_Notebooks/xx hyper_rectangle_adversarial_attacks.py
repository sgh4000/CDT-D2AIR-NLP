
import tensorflow as tf
import numpy as np
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt

class HyperRectangleAdversarialAttacker:
    """
    Advanced adversarial attacker using hyper-rectangle constraints instead of epsilon balls.

    Hyper-rectangles (also called L∞-norm boxes) provide more flexible perturbation
    constraints by allowing different maximum perturbations for each dimension/feature.
    This is particularly useful for text embeddings where different dimensions may
    have different importance or acceptable perturbation ranges.
    """

    def __init__(self, model: tf.keras.Model):
        """
        Initialize the hyper-rectangle adversarial attacker

        Parameters:
        model: Trained TensorFlow model for classification
        """
        self.model = model

    def create_hyper_rectangle_bounds(self, 
                                    embeddings: np.ndarray,
                                    method: str = 'uniform',
                                    base_epsilon: float = 0.01,
                                    dimension_weights: Optional[np.ndarray] = None,
                                    percentile_bounds: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create hyper-rectangle bounds for each dimension

        Parameters:
        embeddings: Original embeddings (batch_size, embedding_dim)
        method: Method for creating bounds ('uniform', 'adaptive', 'weighted', 'percentile')
        base_epsilon: Base perturbation magnitude
        dimension_weights: Weights for each dimension (for 'weighted' method)
        percentile_bounds: Tuple of (lower_percentile, upper_percentile) for 'percentile' method

        Returns:
        lower_bounds, upper_bounds: Arrays defining the hyper-rectangle for each sample
        """
        batch_size, embedding_dim = embeddings.shape

        if method == 'uniform':
            # Standard L∞ ball - same perturbation limit for all dimensions
            epsilon_matrix = np.full((batch_size, embedding_dim), base_epsilon)

        elif method == 'adaptive':
            # Adaptive bounds based on embedding magnitude
            # Larger values get larger perturbation allowances
            embedding_abs = np.abs(embeddings)
            epsilon_matrix = base_epsilon * (1 + embedding_abs / np.mean(embedding_abs))

        elif method == 'weighted':
            # Dimension-specific weights (e.g., based on feature importance)
            if dimension_weights is None:
                dimension_weights = np.ones(embedding_dim)
            epsilon_matrix = base_epsilon * dimension_weights[np.newaxis, :]
            epsilon_matrix = np.repeat(epsilon_matrix, batch_size, axis=0)

        elif method == 'percentile':
            # Bounds based on data distribution percentiles
            if percentile_bounds is None:
                percentile_bounds = (10, 90)

            lower_pct, upper_pct = percentile_bounds
            dim_percentiles = np.percentile(embeddings, [lower_pct, upper_pct], axis=0)

            # Use percentile spread as epsilon for each dimension
            epsilon_per_dim = (dim_percentiles[1] - dim_percentiles[0]) * base_epsilon
            epsilon_matrix = np.tile(epsilon_per_dim, (batch_size, 1))

        else:
            raise ValueError(f"Unknown method: {method}")

        # Calculate bounds
        lower_bounds = embeddings - epsilon_matrix
        upper_bounds = embeddings + epsilon_matrix

        return lower_bounds, upper_bounds

    def project_to_hyper_rectangle(self, 
                                 adversarial_embeddings: tf.Tensor,
                                 lower_bounds: tf.Tensor,
                                 upper_bounds: tf.Tensor) -> tf.Tensor:
        """
        Project adversarial examples back to the hyper-rectangle constraints

        Parameters:
        adversarial_embeddings: Current adversarial embeddings
        lower_bounds: Lower bounds for each dimension
        upper_bounds: Upper bounds for each dimension

        Returns:
        projected_embeddings: Embeddings projected to hyper-rectangle
        """
        return tf.clip_by_value(adversarial_embeddings, lower_bounds, upper_bounds)

    def fgsm_hyper_rectangle(self, 
                           embeddings: np.ndarray, 
                           labels: np.ndarray,
                           bounds_method: str = 'uniform',
                           base_epsilon: float = 0.01,
                           **bounds_kwargs) -> np.ndarray:
        """
        FGSM attack with hyper-rectangle constraints

        Parameters:
        embeddings: Original embeddings
        labels: True labels
        bounds_method: Method for creating hyper-rectangle bounds
        base_epsilon: Base perturbation magnitude
        bounds_kwargs: Additional arguments for bounds creation

        Returns:
        adversarial_embeddings: Adversarial embeddings within hyper-rectangle
        """
        # Create hyper-rectangle bounds
        lower_bounds, upper_bounds = self.create_hyper_rectangle_bounds(
            embeddings, bounds_method, base_epsilon, **bounds_kwargs
        )

        # Convert to tensors
        embeddings_tf = tf.Variable(embeddings, dtype=tf.float32)
        labels_tf = tf.convert_to_tensor(labels, dtype=tf.int64)
        lower_bounds_tf = tf.convert_to_tensor(lower_bounds, dtype=tf.float32)
        upper_bounds_tf = tf.convert_to_tensor(upper_bounds, dtype=tf.float32)

        # Compute gradients
        with tf.GradientTape() as tape:
            predictions = self.model(embeddings_tf)
            loss = tf.keras.losses.sparse_categorical_crossentropy(labels_tf, predictions)
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, embeddings_tf)

        # FGSM step with dimension-specific epsilon
        perturbation_magnitude = upper_bounds_tf - embeddings_tf
        perturbation = perturbation_magnitude * tf.sign(gradients)
        adversarial_embeddings = embeddings_tf + perturbation

        # Project to hyper-rectangle
        adversarial_embeddings = self.project_to_hyper_rectangle(
            adversarial_embeddings, lower_bounds_tf, upper_bounds_tf
        )

        return adversarial_embeddings.numpy(), (lower_bounds, upper_bounds)

    def pgd_hyper_rectangle(self,
                          embeddings: np.ndarray,
                          labels: np.ndarray,
                          bounds_method: str = 'uniform',
                          base_epsilon: float = 0.01,
                          alpha_ratio: float = 0.1,
                          num_iter: int = 10,
                          random_start: bool = True,
                          **bounds_kwargs) -> np.ndarray:
        """
        PGD attack with hyper-rectangle constraints

        Parameters:
        embeddings: Original embeddings
        labels: True labels
        bounds_method: Method for creating hyper-rectangle bounds
        base_epsilon: Base perturbation magnitude
        alpha_ratio: Step size as ratio of perturbation magnitude
        num_iter: Number of PGD iterations
        random_start: Whether to start with random perturbation
        bounds_kwargs: Additional arguments for bounds creation

        Returns:
        adversarial_embeddings: Final adversarial embeddings
        """
        # Create hyper-rectangle bounds
        lower_bounds, upper_bounds = self.create_hyper_rectangle_bounds(
            embeddings, bounds_method, base_epsilon, **bounds_kwargs
        )

        # Convert to tensors
        embeddings_tf = tf.convert_to_tensor(embeddings, dtype=tf.float32)
        labels_tf = tf.convert_to_tensor(labels, dtype=tf.int64)
        lower_bounds_tf = tf.convert_to_tensor(lower_bounds, dtype=tf.float32)
        upper_bounds_tf = tf.convert_to_tensor(upper_bounds, dtype=tf.float32)

        # Initialize adversarial examples
        if random_start:
            # Random start within hyper-rectangle
            random_noise = tf.random.uniform(embeddings_tf.shape, -1.0, 1.0)
            perturbation_range = upper_bounds_tf - lower_bounds_tf
            initial_noise = random_noise * perturbation_range * 0.5
            adversarial_embeddings = embeddings_tf + initial_noise
        else:
            adversarial_embeddings = tf.identity(embeddings_tf)

        # Project initial point to hyper-rectangle
        adversarial_embeddings = self.project_to_hyper_rectangle(
            adversarial_embeddings, lower_bounds_tf, upper_bounds_tf
        )

        # PGD iterations
        for iteration in range(num_iter):
            adversarial_embeddings = tf.Variable(adversarial_embeddings)

            with tf.GradientTape() as tape:
                predictions = self.model(adversarial_embeddings)
                loss = tf.keras.losses.sparse_categorical_crossentropy(labels_tf, predictions)
                loss = tf.reduce_mean(loss)

            gradients = tape.gradient(loss, adversarial_embeddings)

            # PGD step with dimension-specific alpha
            perturbation_range = upper_bounds_tf - lower_bounds_tf
            alpha = alpha_ratio * perturbation_range
            adversarial_embeddings = adversarial_embeddings + alpha * tf.sign(gradients)

            # Project back to hyper-rectangle
            adversarial_embeddings = self.project_to_hyper_rectangle(
                adversarial_embeddings, lower_bounds_tf, upper_bounds_tf
            )

        return adversarial_embeddings.numpy(), (lower_bounds, upper_bounds)

    def analyze_hyper_rectangle_attack(self,
                                     original_embeddings: np.ndarray,
                                     adversarial_embeddings: np.ndarray,
                                     bounds: Tuple[np.ndarray, np.ndarray],
                                     labels: np.ndarray) -> dict:
        """
        Analyze the effectiveness of hyper-rectangle attacks

        Parameters:
        original_embeddings: Original embeddings
        adversarial_embeddings: Adversarial embeddings
        bounds: Tuple of (lower_bounds, upper_bounds)
        labels: True labels

        Returns:
        analysis_results: Dictionary containing analysis metrics
        """
        lower_bounds, upper_bounds = bounds

        # Get predictions
        original_preds = self.model.predict(original_embeddings, verbose=0)
        adversarial_preds = self.model.predict(adversarial_embeddings, verbose=0)

        original_classes = np.argmax(original_preds, axis=1)
        adversarial_classes = np.argmax(adversarial_preds, axis=1)

        # Calculate metrics
        original_accuracy = np.mean(original_classes == labels)
        adversarial_accuracy = np.mean(adversarial_classes == labels)
        attack_success_rate = np.mean(original_classes != adversarial_classes)

        # Perturbation analysis
        perturbations = adversarial_embeddings - original_embeddings

        # L∞ norm (should be within bounds)
        linf_norms = np.max(np.abs(perturbations), axis=1)

        # L2 norm
        l2_norms = np.linalg.norm(perturbations, axis=1)

        # L0 norm (sparsity)
        l0_norms = np.sum(np.abs(perturbations) > 1e-8, axis=1)

        # Bound utilization analysis
        perturbation_ranges = upper_bounds - lower_bounds
        bound_utilization = np.abs(perturbations) / perturbation_ranges
        avg_bound_utilization = np.mean(bound_utilization)

        # Per-dimension analysis
        dim_perturbation_stats = {
            'mean_abs_perturbation': np.mean(np.abs(perturbations), axis=0),
            'max_abs_perturbation': np.max(np.abs(perturbations), axis=0),
            'perturbation_std': np.std(perturbations, axis=0)
        }

        analysis_results = {
            'original_accuracy': original_accuracy,
            'adversarial_accuracy': adversarial_accuracy,
            'attack_success_rate': attack_success_rate,
            'mean_l2_norm': np.mean(l2_norms),
            'mean_linf_norm': np.mean(linf_norms),
            'mean_l0_norm': np.mean(l0_norms),
            'bound_utilization': avg_bound_utilization,
            'perturbation_ranges': perturbation_ranges,
            'dimension_stats': dim_perturbation_stats,
            'per_sample_metrics': {
                'l2_norms': l2_norms,
                'linf_norms': linf_norms,
                'l0_norms': l0_norms,
                'bound_utilization_per_sample': np.mean(bound_utilization, axis=1)
            }
        }

        return analysis_results

    def visualize_hyper_rectangle_attack(self,
                                       analysis_results: dict,
                                       save_path: str = None):
        """
        Create visualizations for hyper-rectangle attack analysis
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Hyper-Rectangle Adversarial Attack Analysis', fontsize=16, fontweight='bold')

        # 1. Attack success metrics
        metrics = ['Original Accuracy', 'Adversarial Accuracy', 'Attack Success Rate']
        values = [analysis_results['original_accuracy'], 
                 analysis_results['adversarial_accuracy'],
                 analysis_results['attack_success_rate']]

        axes[0, 0].bar(metrics, values)
        axes[0, 0].set_title('Attack Success Metrics')
        axes[0, 0].set_ylabel('Rate')
        axes[0, 0].set_ylim(0, 1)

        # 2. Perturbation norms
        per_sample = analysis_results['per_sample_metrics']
        axes[0, 1].hist(per_sample['l2_norms'], bins=30, alpha=0.7, label='L2', color='blue')
        axes[0, 1].hist(per_sample['linf_norms'], bins=30, alpha=0.7, label='L∞', color='red')
        axes[0, 1].set_title('Perturbation Magnitude Distribution')
        axes[0, 1].set_xlabel('Perturbation Magnitude')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()

        # 3. L0 norm (sparsity)
        axes[0, 2].hist(per_sample['l0_norms'], bins=30, alpha=0.7, color='green')
        axes[0, 2].set_title('Attack Sparsity (L0 Norm)')
        axes[0, 2].set_xlabel('Number of Perturbed Dimensions')
        axes[0, 2].set_ylabel('Frequency')

        # 4. Bound utilization
        axes[1, 0].hist(per_sample['bound_utilization_per_sample'], bins=30, alpha=0.7, color='orange')
        axes[1, 0].set_title('Hyper-Rectangle Bound Utilization')
        axes[1, 0].set_xlabel('Average Bound Utilization')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(analysis_results['bound_utilization'], color='red', linestyle='--', 
                          label=f'Mean: {analysis_results["bound_utilization"]:.3f}')
        axes[1, 0].legend()

        # 5. Per-dimension perturbation
        dim_stats = analysis_results['dimension_stats']
        dimensions = np.arange(len(dim_stats['mean_abs_perturbation']))[:50]  # Show first 50 dims
        axes[1, 1].plot(dimensions, dim_stats['mean_abs_perturbation'][:50], 'b-', alpha=0.7)
        axes[1, 1].fill_between(dimensions, 
                               dim_stats['mean_abs_perturbation'][:50] - dim_stats['perturbation_std'][:50],
                               dim_stats['mean_abs_perturbation'][:50] + dim_stats['perturbation_std'][:50],
                               alpha=0.3)
        axes[1, 1].set_title('Per-Dimension Perturbation (First 50 Dims)')
        axes[1, 1].set_xlabel('Embedding Dimension')
        axes[1, 1].set_ylabel('Mean Absolute Perturbation')

        # 6. Perturbation vs bounds correlation
        if len(analysis_results['perturbation_ranges'].shape) == 2:
            mean_ranges = np.mean(analysis_results['perturbation_ranges'], axis=0)[:50]
            axes[1, 2].scatter(mean_ranges, dim_stats['mean_abs_perturbation'][:50], alpha=0.6)
            axes[1, 2].set_xlabel('Perturbation Range (Bound Width)')
            axes[1, 2].set_ylabel('Mean Absolute Perturbation')
            axes[1, 2].set_title('Perturbation vs Bound Width Correlation')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Analysis plots saved to {save_path}")

        plt.show()


def compare_epsilon_ball_vs_hyper_rectangle():
    """
    Comparison function to demonstrate differences between epsilon balls and hyper-rectangles
    """
    print("Epsilon Ball vs Hyper-Rectangle Constraints")
    print("=" * 50)

    comparison = {
        "Epsilon Ball (L2/L∞ norm)": {
            "Shape": "Sphere (L2) or Hypercube (L∞) with uniform constraints",
            "Flexibility": "Low - same perturbation limit for all dimensions",
            "Use case": "General adversarial attacks, theoretical analysis",
            "Advantages": ["Simple to implement", "Mathematically clean", "Well-studied"],
            "Disadvantages": ["May be too restrictive", "Doesn't account for feature importance", 
                            "Uniform constraints may be suboptimal"]
        },
        "Hyper-Rectangle": {
            "Shape": "Rectangular box with dimension-specific constraints",
            "Flexibility": "High - different perturbation limits per dimension",
            "Use case": "Feature-aware attacks, text embeddings, specialized domains",
            "Advantages": ["Dimension-specific control", "Can respect feature importance", 
                          "More realistic constraints", "Better semantic preservation"],
            "Disadvantages": ["More complex to implement", "Requires domain knowledge", 
                             "More hyperparameters"]
        }
    }

    for method, details in comparison.items():
        print(f"\n{method}:")
        for key, value in details.items():
            if isinstance(value, list):
                print(f"  {key}: {', '.join(value)}")
            else:
                print(f"  {key}: {value}")

    print("\n" + "=" * 50)
    print("Recommendation for PhD Research:")
    print("Use hyper-rectangles when:")
    print("• Working with high-dimensional embeddings (like sentence embeddings)")
    print("• Different dimensions have different semantic importance")
    print("• You want more control over perturbation constraints")
    print("• Studying realistic adversarial scenarios")


# Example usage demonstration
def demonstrate_hyper_rectangle_attack():
    """
    Demonstrate hyper-rectangle attacks on dummy data
    """
    print("Demonstrating Hyper-Rectangle Adversarial Attacks")
    print("=" * 55)

    # Create dummy model and data
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(384,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Dummy training
    X_train = np.random.randn(1000, 384)
    y_train = np.random.randint(0, 3, 1000)
    model.fit(X_train, y_train, epochs=3, verbose=0)

    # Test data
    X_test = np.random.randn(100, 384)
    y_test = np.random.randint(0, 3, 100)

    # Initialize attacker
    attacker = HyperRectangleAdversarialAttacker(model)

    # Demonstrate different bound methods
    methods = ['uniform', 'adaptive', 'weighted']

    for method in methods:
        print(f"\nTesting {method} bounds method:")

        if method == 'weighted':
            # Create example dimension weights (higher for first 50 dimensions)
            dim_weights = np.ones(384)
            dim_weights[:50] = 2.0  # More important dimensions get larger perturbation allowance

            adv_embeddings, bounds = attacker.fgsm_hyper_rectangle(
                X_test[:20], y_test[:20], bounds_method=method, 
                base_epsilon=0.01, dimension_weights=dim_weights
            )
        else:
            adv_embeddings, bounds = attacker.fgsm_hyper_rectangle(
                X_test[:20], y_test[:20], bounds_method=method, base_epsilon=0.01
            )

        # Analyze results
        analysis = attacker.analyze_hyper_rectangle_attack(
            X_test[:20], adv_embeddings, bounds, y_test[:20]
        )

        print(f"  Attack success rate: {analysis['attack_success_rate']:.2%}")
        print(f"  Mean L2 perturbation: {analysis['mean_l2_norm']:.6f}")
        print(f"  Mean L∞ perturbation: {analysis['mean_linf_norm']:.6f}")
        print(f"  Bound utilization: {analysis['bound_utilization']:.2%}")

if __name__ == "__main__":
    compare_epsilon_ball_vs_hyper_rectangle()
    print("\n" + "=" * 70)
    demonstrate_hyper_rectangle_attack()
