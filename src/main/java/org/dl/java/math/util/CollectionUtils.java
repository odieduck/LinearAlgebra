package org.dl.java.math.util;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.function.BiFunction;

import com.google.common.base.Preconditions;

/**
 * Contains convenient methods for collections
 */
public class CollectionUtils {

    /**
     * Given two list, iterate through them simultaneously and do the work provided by the function
     *
     * @param listOne
     * @param listTwo
     * @param function
     * @param <T>
     * @param <R>
     * @return
     */
    public static <T, K, R> List<R> zipApply(List<T> listOne, List<K> listTwo, BiFunction<T, K, R> function) {
        Preconditions.checkNotNull(listOne, "input list one cannot be null");
        Preconditions.checkNotNull(listTwo, "input list two cannot be null");
        Preconditions.checkNotNull(function, "the function cannot be null");
        Preconditions.checkArgument(listOne.size() == listTwo.size(), "Two lists must have the same size");

        Iterator<T> i1 = listOne.iterator();
        Iterator<K> i2 = listTwo.iterator();

        List<R> res = new ArrayList<>();
        while (i1.hasNext() && i2.hasNext()) {
            res.add(function.apply(i1.next(), i2.next()));
        }

        return res;
    }
}
