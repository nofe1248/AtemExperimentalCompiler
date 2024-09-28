module;

#include <array>
#include <cassert>
#include <concepts>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

#define FWD(x) ::std::forward<decltype(x)>(x)

export module Atem.Utils.Result;

namespace atem::utils::detail
{
template <typename T, typename U>
using CopyConstness = std::conditional_t<std::is_const_v<T>, const std::remove_const_t<U>, std::remove_const_t<U>>;
}

export namespace atem::utils
{
class Error
{
private:
    std::string what_;

public:
    explicit Error(std::string what) : what_(std::move(what))
    {
    }

    Error(const Error &other) = default;

    Error(Error &&other) noexcept : what_{std::move(other.what_)}
    {
    }

    Error &operator=(const Error &other)
    {
        if (this == &other)
        {
            return *this;
        }
        what_ = other.what_;
        return *this;
    }

    Error &operator=(Error &&other) noexcept
    {
        if (this == &other)
        {
            return *this;
        }
        what_ = std::move(other.what_);
        return *this;
    }

    ~Error() = default;

    [[nodiscard]] auto what() const -> const std::string &
    {
        return this->what_;
    }
};

struct NullResult
{
};

template <typename ResultTypeT = NullResult>
class Result
{
    static_assert(not std::same_as<ResultTypeT, Error>, "The result type cannot be atem::utils::Error");

private:
    bool success_ = false;

    using ResultTypeOrErrorArray = std::array<unsigned char, std::max(sizeof(ResultTypeT), sizeof(Error))>;

    ResultTypeOrErrorArray result_or_error_;

public:
    using ResultType = ResultTypeT;

    explicit Result(ResultType const &result) : success_(true)
    {
        new (&this->getResultTypeFromUnderlyingArray()) ResultType(result);
    }

    explicit Result(ResultType &&result) : success_(true)
    {
        new (&this->getResultTypeFromUnderlyingArray()) ResultType(std::move(result));
    }

    explicit Result(Error const &error) : success_(false)
    {
        new (&this->getErrorTypeFromUnderlyingArray()) Error(error);
    }

    explicit Result(Error &&error) : success_(false)
    {
        new (&this->getErrorTypeFromUnderlyingArray()) Error(std::move(error));
    }

    Result(Result const &that) : success_(that.success_)
    {
        this->copyFromOther(that);
    }

    Result(Result &&that) noexcept : success_(that.success_)
    {
        this->moveFromOther(that);
    }

    template <typename SourceResultTypeT>
        requires std::convertible_to<SourceResultTypeT, ResultTypeT>
    explicit Result(Result<SourceResultTypeT> const &that) : success_(that.success_)
    {
        this->moveFromOther(that.transform([](SourceResultTypeT const &source_result) { return static_cast<ResultTypeT>(source_result); }));
    }

    template <typename SourceResultTypeT>
        requires std::convertible_to<SourceResultTypeT, ResultTypeT>
    explicit Result(Result<SourceResultTypeT> &&that) : success_(that.success_)
    {
        this->moveFromOther(std::forward<Result<SourceResultTypeT>>(that).transform(
            [](SourceResultTypeT &&source_result) { return static_cast<ResultTypeT>(std::forward<SourceResultTypeT>(source_result)); }));
    }

    ~Result()
    {
        this->destroyUnderlyingObject();
    }

    auto operator=(Result<ResultType> const &that) -> Result<ResultType> &
    {
        if (this == &that)
        {
            return *this;
        }
        this->destroyUnderlyingObject();
        this->success_ = that.success_;
        this->copyFromOther(that);
        return *this;
    }

    auto operator=(Result<ResultType> &&that) noexcept -> Result<ResultType> &
    {
        if (this == &that)
        {
            return *this;
        }
        this->destroyUnderlyingObject();
        this->success_ = that.success_;
        this->moveFromOther(that);
        return *this;
    }

    explicit operator bool() const noexcept
    {
        return this->success_;
    }

    auto operator*(this auto &&self) noexcept -> detail::CopyConstness<decltype(self), ResultType> &
    {
        assert(FWD(self).success_);
        return FWD(self).getResultTypeFromUnderlyingArray();
    }

    [[nodiscard]] auto error() const noexcept -> std::optional<Error>
    {
        if (this->success_)
        {
            return std::nullopt;
        }
        return this->getErrorTypeFromUnderlyingArray();
    }

    [[nodiscard]] auto value() const noexcept -> std::optional<ResultType>
    {
        if (this->success_)
        {
            return this->getResultTypeFromUnderlyingArray();
        }
        return std::nullopt;
    }

    [[nodiscard]] auto valueOr(ResultType &&default_value) noexcept -> ResultType
    {
        if (this->success_)
        {
            return std::forward<ResultType>(this->getResultTypeFromUnderlyingArray());
        }
        return std::forward<ResultType>(default_value);
    }

    [[nodiscard]] auto valueOr(ResultType const &default_value) const noexcept -> ResultType
    {
        if (this->success_)
        {
            return this->getResultTypeFromUnderlyingArray();
        }
        return default_value;
    }

    template <typename FuncT>
    auto andThen(this auto &&self, FuncT const &func) -> std::invoke_result_t<FuncT, ResultType>
    {
        using ReturnType = std::invoke_result_t<FuncT, ResultType>;
        if constexpr (std::is_const_v<decltype(self)>)
        {
            if (FWD(self).success_)
            {
                return ReturnType(func(FWD(self).getResultTypeFromUnderlyingArray()));
            }
            return ReturnType(FWD(self).getErrorTypeFromUnderlyingArray());
        }
        else
        {
            if (FWD(self).success_)
            {
                return ReturnType(func(std::forward<ResultType>(FWD(self).getResultTypeFromUnderlyingArray())));
            }
            return ReturnType(std::forward<Error>(FWD(self).getErrorTypeFromUnderlyingArray()));
        }
    }

    template <typename FuncT>
    auto transform(this auto &&self, FuncT const &func) -> std::invoke_result_t<FuncT, ResultType>
    {
        using ReturnType = std::invoke_result_t<FuncT, ResultType>;
        if constexpr (std::is_const_v<decltype(self)>)
        {
            if (FWD(self).success_)
            {
                return Result<ReturnType>(func(FWD(self).getResultTypeFromUnderlyingArray()));
            }
            return Result<ReturnType>(FWD(self).getErrorTypeFromUnderlyingArray());
        }
        else
        {
            if (FWD(self).success_)
            {
                return Result<ReturnType>(func(std::forward<ResultType>(FWD(self).getResultTypeFromUnderlyingArray())));
            }
            return Result<ReturnType>(std::forward<Error>(FWD(self).getErrorTypeFromUnderlyingArray()));
        }
    }

    auto begin() const noexcept -> const ResultType *
    {
        if (this->success_)
        {
            return this->getResultTypeFromUnderlyingArray();
        }
        return nullptr;
    }

    auto begin() noexcept -> ResultType *
    {
        if (this->success_)
        {
            return this->getResultTypeFromUnderlyingArray();
        }
        return nullptr;
    }

    auto end() const noexcept -> const ResultType *
    {
        if (this->success_)
        {
            return this->getResultTypeFromUnderlyingArray();
        }
        return nullptr;
    }

    auto end() noexcept -> ResultType *
    {
        if (this->success_)
        {
            return this->getResultTypeFromUnderlyingArray();
        }
        return nullptr;
    }

private:
    [[nodiscard]] auto getResultTypeFromUnderlyingArray() noexcept -> ResultType &
    {
        return *(reinterpret_cast<ResultType *>(this->result_or_error_.data()));
    }

    [[nodiscard]] auto getResultTypeFromUnderlyingArray() const noexcept -> ResultType const &
    {
        return *(reinterpret_cast<ResultType *>(this->result_or_error_.data()));
    }

    [[nodiscard]] auto getErrorTypeFromUnderlyingArray() noexcept -> Error &
    {
        return *(reinterpret_cast<Error *>(this->result_or_error_.data()));
    }

    [[nodiscard]] auto getErrorTypeFromUnderlyingArray() const noexcept -> Error const &
    {
        return *(reinterpret_cast<Error *>(this->result_or_error_.data()));
    }

    auto destroyUnderlyingObject() -> void
    {
        if (this->success_)
        {
            if constexpr (std::destructible<ResultType>)
            {
                this->getResultTypeFromUnderlyingArray().~ResultType();
            }
        }
        else
        {
            this->getErrorTypeFromUnderlyingArray().~Error();
        }
    }

    auto copyFromOther(Result const &that) -> void
    {
        if (this->success_)
        {
            new (&this->getResultTypeFromUnderlyingArray()) Result(that.getResultTypeFromUnderlyingArray());
        }
        else
        {
            new (&this->getErrorTypeFromUnderlyingArray()) Error(that.getErrorTypeFromUnderlyingArray());
        }
    }

    auto moveFromOther(Result &&that) -> void
    {
        if (this->success_)
        {
            new (&this->getResultTypeFromUnderlyingArray()) Result(std::move(that.getResultTypeFromUnderlyingArray()));
        }
        else
        {
            new (&this->getErrorTypeFromUnderlyingArray()) Error(std::move(that.getErrorTypeFromUnderlyingArray()));
        }
    }
};
} // namespace atem::utils
