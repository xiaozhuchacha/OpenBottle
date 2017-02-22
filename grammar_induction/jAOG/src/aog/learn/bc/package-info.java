/**
 * The code for learning AOG from a training corpus by bottom-up structure search
 * using iterative biclustering. The learned AOG is in CNF. This implementation 
 * employs a lot of caching and is more efficient than the code in aog.learn.greedy.
 * It however does not support noisy relations and non-default combiners (e.g., anchor
 * point displacement in image AOG).
 */
/**
 * @author Kewei Tu
 *
 */
package aog.learn.bc;