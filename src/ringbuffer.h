#ifndef _RING_BUFFER_H_
#define _RING_BUFFER_H_

///
/// @class RingBuffer
/// @brief Implementing a ringbuffer, where new elements are added to tail and older are removed from head.
///
template<typename T, unsigned int kMaxCapacity>
class RingBuffer
{
  public:
    typedef unsigned int SizeType;
    typedef T ValueType;
    typedef ValueType& Reference;
    typedef const ValueType& ConstReference;
    
    static const SizeType kCapacityLimit = kMaxCapacity;

    explicit RingBuffer(SizeType capacity = kMaxCapacity)
        :head_(0U), tail_(0U), size_(0U), capacity_(capacity)
    {
        SetMaxSize(capacity);
    }

    ~RingBuffer()
    {
    }

    bool IsEmpty() const
    {
        return size_ == 0;
    }
   
    Reference Front(bool& result)
    {
        return GetArrayElement(GetArrayIndexForFront(), result);
    }

    Reference Back(bool& result)
    {
        return GetArrayElement(GetArrayIndexForBack(), result);
    }

    ///
    /// @return const reference to the oldest element. 
    ///
    /// @param[out] result  true if element was found at location, false if out of bounds
    ///
    ConstReference Front(bool& result) const
    {
        return GetArrayElement(GetArrayIndexForFront(), result);
    }

    ///
    /// @return Const reference to the youngest element in the buffer
    ///
    /// @param[out] result  true if element was found at location, false if out of bounds
    ///
    inline ConstReference Back(bool& result) const
    {
        return GetArrayElement(GetArrayIndexForBack(), result);
    }
   
    void PushBack(ConstReference value);

    ///
    /// @brief  Removes the oldest element from the buffer. No effect if called on empty buffer.
    ///
    bool PopFront();

    ///
    /// @brief Removes the youngest element from the buffer. No effect if called on empty buffer.
    ///
    bool PopBack();

    ///
    /// @brief  returns size of the buffer
    ///
    SizeType Size() const
    {
        return size_;
    }

    ///
    /// @brief  empties the buffer by resetting it to the initial state. 
    ///
    /// @note Does not call destructors on objects!
    ///
    void Clear()
    {
        head_ = 0U;
        tail_ = 0U;
        size_ = 0U;
    }
    
    ///
    /// @brief  retrieves buffer element by index. Performs no bounds check.
    /// @param[in] index    Index is zero-based, where zero is the oldest element. The newest is then [Size() - 1]
    /// @param[out] result  true if element was found at location, false if out of bounds
    ///
    ConstReference GetElementAt(SizeType index, bool& result) const
    {
        return GetArrayElement(GetArrayElementIndex(index), result);
    }
    
    ///
    /// @brief  retrieves buffer element by index. Performs no bounds check.
    /// 
    /// @param[in] index    Index is zero-based, where zero is the oldest element. The newest is then [Size() - 1]
    /// @param[out] result  true if element was found at location, false if out of bounds
    ///
    Reference GetElementAt(SizeType index, bool& result)
    {
        return GetArrayElement(GetArrayElementIndex(index), result);
    }

    ///
    /// @return maximum capacity of the buffer. The buffer size may not exceed this value.
    ///
    SizeType MaxSize() const
    {
        return capacity_;
    }

    ///
    /// @brief  Clears the buffer and sets new buffer capacity. 
    ///
    /// @param  new_max_size new size of the buffer
    void SetMaxSize(SizeType new_max_size);

  private:

    SizeType GetIndexAdjustedForBoundaries(SizeType index) const
    {
        return index % capacity_;
    }

    ///
    /// @brief Adds 1 to the index, wrapping around if necessary
    ///
    SizeType Increment(SizeType index) const
    {
        return GetIndexAdjustedForBoundaries(index + 1);
    }

    ///
    /// @brief Subtracts 1 to the index, wrapping around if necessary.
    ///
    SizeType Decrement(SizeType index) const
    {
        SizeType decremented_index = 0U;

        if (index != 0U)
        {
            decremented_index = index-1;
        }
        else
        {
            decremented_index = capacity_-1;
        }

        return decremented_index;
    }

    SizeType GetArrayIndexForBack() const
    {
        if (IsEmpty())
        {
            return kCapacityLimit;
        }
        else
        {
            return GetIndexAdjustedForBoundaries(capacity_ - 1 + tail_);
        }        
    }

    SizeType GetArrayIndexForFront() const
    {
        if (IsEmpty())
        {
            return kCapacityLimit;
        }
        else
        {
            return head_;
        }
    }

    SizeType GetArrayElementIndex(const SizeType relative_index) const;
    Reference GetArrayElement(SizeType index, bool& result);
    ConstReference GetArrayElement(SizeType index, bool& result) const;
   
    SizeType head_;
    SizeType tail_;
    SizeType size_;
    SizeType capacity_;

    ValueType data_[kMaxCapacity];
};

template <typename T, unsigned int kMaxCapacity>
void RingBuffer<T, kMaxCapacity>::PushBack(ConstReference value)
{
    //tail reached the head. Drop one element if necessary.
    if (size_ == capacity_)
    {
        PopFront();
    }
    data_[tail_] = value;
    tail_ = Increment(tail_);
    size_++;
}

template <typename T, unsigned int kMaxCapacity>
bool RingBuffer<T, kMaxCapacity>::PopFront()
{
    bool result = false;
    // If there are no elements, just return.
    if (!IsEmpty())
    {
        head_ = Increment(head_);
        size_--;
        result = true;
    }
    return result;
}

template <typename T, unsigned int kMaxCapacity>
bool RingBuffer<T, kMaxCapacity>::PopBack()
{
    bool result = false;
    // If there are no elements, just return.
    if (!IsEmpty())
    {
        tail_ = Decrement(tail_);
        size_--;
        result = true;
    }
    return result;
}

template <typename T, unsigned int kMaxCapacity>
void RingBuffer<T, kMaxCapacity>::SetMaxSize(SizeType new_max_size)
{
    if (new_max_size > kMaxCapacity)
    {
        new_max_size = kMaxCapacity;
    }
    if (new_max_size == 0)
    {
        new_max_size = 1U;
    }
    Clear();
    capacity_ = new_max_size;
}

template <typename T, unsigned int kMaxCapacity>
typename RingBuffer<T, kMaxCapacity>::SizeType RingBuffer<T, kMaxCapacity>::GetArrayElementIndex(const SizeType relative_index) const
{
    SizeType act_index = kCapacityLimit;
    if (relative_index < size_)
    {
        act_index = GetIndexAdjustedForBoundaries(head_ + relative_index);
    }
    return act_index;
}

template <typename T, unsigned int kMaxCapacity>
typename RingBuffer<T, kMaxCapacity>::Reference RingBuffer<T, kMaxCapacity>::GetArrayElement(SizeType index, bool& result)
{
    if (index >= capacity_)
    {
        index = 0U;
        result = false;
    }
    else
    {
        result = true;
    }
    return data_[index];
}

template <typename T, unsigned int kMaxCapacity>
typename RingBuffer<T, kMaxCapacity>::ConstReference RingBuffer<T, kMaxCapacity>::GetArrayElement(SizeType index, bool& result) const
{
    if (index >= capacity_)
    {
        index = 0U;
        result = false;
    }
    else
    {
        result = true;
    }
    return data_[index];
}


#endif