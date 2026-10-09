#include <mpi.h>
#include <omp.h>

#ifdef USE_CUDA
#include "CudaCosineSimilarity.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cctype>
#include <climits>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {
struct Options {
    std::string input = "movies_metadata - Sheet1.csv";
    std::string stopwords = "StopWords.txt";
    std::string output = "results/top_matches.csv";
    std::size_t limit = 1000;
    std::size_t vocabulary = 256;
    std::size_t top = 20;
    bool verify = false;
};

struct Movie {
    std::string title;
    std::string description;
};

struct Counts {
    std::unordered_map<std::string, int> terms;
    int total = 0;
};

struct Match {
    std::size_t first;
    std::size_t second;
    double score;
};

std::size_t positiveNumber(const std::string& text, const std::string& flag) {
    std::size_t used = 0;
    const unsigned long long number = std::stoull(text, &used);
    if (used != text.size() || number == 0 ||
        number > static_cast<unsigned long long>(INT_MAX)) {
        throw std::invalid_argument(flag + " must be a positive integer up to INT_MAX");
    }
    return static_cast<std::size_t>(number);
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--verify") {
            options.verify = true;
            continue;
        }
        if (i + 1 == argc) throw std::invalid_argument("missing value for " + flag);
        const std::string value = argv[++i];
        if (flag == "--input") options.input = value;
        else if (flag == "--stopwords") options.stopwords = value;
        else if (flag == "--output") options.output = value;
        else if (flag == "--limit") options.limit = positiveNumber(value, flag);
        else if (flag == "--vocab") options.vocabulary = positiveNumber(value, flag);
        else if (flag == "--top") options.top = positiveNumber(value, flag);
        else throw std::invalid_argument("unknown option: " + flag);
    }
    return options;
}

// Parse quoted fields, doubled quotes, and embedded newlines.
bool readCsvRecord(std::istream& input, std::vector<std::string>& fields) {
    fields.clear();
    std::string field;
    bool quoted = false;
    bool fieldStart = true;
    bool any = false;
    char character;
    while (input.get(character)) {
        any = true;
        if (quoted) {
            if (character == '"' && input.peek() == '"') {
                input.get(character);
                field += '"';
            } else if (character == '"') {
                quoted = false;
            } else {
                field += character;
            }
        } else if (character == '"' && fieldStart) {
            quoted = true;
            fieldStart = false;
        } else if (character == ',') {
            fields.push_back(field);
            field.clear();
            fieldStart = true;
        } else if (character == '\n') {
            if (!field.empty() && field.back() == '\r') field.pop_back();
            fields.push_back(field);
            return true;
        } else {
            field += character;
            fieldStart = false;
        }
    }
    if (quoted) throw std::runtime_error("unterminated quoted CSV field");
    if (!any) return false;
    if (!field.empty() && field.back() == '\r') field.pop_back();
    fields.push_back(field);
    return true;
}

std::vector<Movie> readMovies(const Options& options) {
    std::ifstream input(options.input, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open input: " + options.input);
    std::vector<Movie> movies;
    std::vector<std::string> fields;
    while (movies.size() < options.limit && readCsvRecord(input, fields)) {
        if (fields.size() >= 2 && !fields[0].empty() && !fields[1].empty()) {
            movies.push_back({fields[0], fields[1]});
        }
    }
    if (movies.size() < 2) throw std::runtime_error("input needs at least two movies with descriptions");
    return movies;
}

std::unordered_set<std::string> readStopwords(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open stop-word file: " + path);
    std::unordered_set<std::string> words;
    std::string word;
    while (input >> word) words.insert(word);
    return words;
}

Counts countTerms(const std::string& text,
                  const std::unordered_set<std::string>& stopwords) {
    Counts result;
    std::string word;
    auto addWord = [&]() {
        if (!word.empty() && !stopwords.count(word)) {
            ++result.terms[word];
            ++result.total;
        }
        word.clear();
    };
    for (unsigned char character : text) {
        if (character < 128 && std::isalnum(character)) {
            word += static_cast<char>(std::tolower(character));
        } else {
            addWord();
        }
    }
    addWord();
    return result;
}

std::vector<Counts> gatherCounts(const std::vector<Movie>& movies,
                                 const std::unordered_set<std::string>& stopwords,
                                 int rank, int tasks) {
    const std::size_t begin = movies.size() * static_cast<std::size_t>(rank) / tasks;
    const std::size_t end = movies.size() * static_cast<std::size_t>(rank + 1) / tasks;
    std::vector<Counts> local(end - begin);
#pragma omp parallel for schedule(dynamic)
    for (long long offset = 0; offset < static_cast<long long>(local.size()); ++offset) {
        local[static_cast<std::size_t>(offset)] =
            countTerms(movies[begin + static_cast<std::size_t>(offset)].description, stopwords);
    }

    std::ostringstream serialized;
    for (std::size_t offset = 0; offset < local.size(); ++offset) {
        const std::size_t id = begin + offset;
        serialized << "D " << id << ' ' << local[offset].total << '\n';
        for (const auto& term : local[offset].terms) {
            serialized << "T " << id << ' ' << term.first << ' ' << term.second << '\n';
        }
    }
    const std::string payload = serialized.str();
    if (payload.size() > static_cast<std::size_t>(INT_MAX)) {
        throw std::runtime_error("worker payload exceeds MPI int count range");
    }
    const int sendSize = static_cast<int>(payload.size());
    std::vector<int> sizes(rank == 0 ? tasks : 0);
    MPI_Gather(&sendSize, 1, MPI_INT, rank == 0 ? sizes.data() : nullptr,
               1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> offsets(rank == 0 ? tasks : 0);
    std::vector<char> gathered;
    if (rank == 0) {
        long long total = 0;
        for (int task = 0; task < tasks; ++task) {
            offsets[task] = static_cast<int>(total);
            total += sizes[task];
            if (total > INT_MAX) throw std::runtime_error("gather exceeds MPI int count range");
        }
        gathered.resize(static_cast<std::size_t>(total));
    }
    MPI_Gatherv(payload.data(), sendSize, MPI_CHAR,
                rank == 0 ? gathered.data() : nullptr,
                rank == 0 ? sizes.data() : nullptr,
                rank == 0 ? offsets.data() : nullptr,
                MPI_CHAR, 0, MPI_COMM_WORLD);
    if (rank != 0) return {};

    std::vector<Counts> counts(movies.size());
    std::vector<bool> seen(movies.size(), false);
    std::istringstream input(std::string(gathered.begin(), gathered.end()));
    char type;
    std::size_t id;
    while (input >> type >> id) {
        if (id >= movies.size()) throw std::runtime_error("invalid gathered document ID");
        if (type == 'D') {
            input >> counts[id].total;
            seen[id] = true;
        } else if (type == 'T') {
            std::string word;
            int occurrences;
            input >> word >> occurrences;
            counts[id].terms[word] = occurrences;
        } else {
            throw std::runtime_error("invalid gathered record");
        }
        if (!input) throw std::runtime_error("truncated gathered record");
    }
    if (std::find(seen.begin(), seen.end(), false) != seen.end()) {
        throw std::runtime_error("missing gathered document");
    }
    return counts;
}

std::vector<double> makeVectors(const std::vector<Counts>& counts,
                                std::size_t requestedVocabulary, std::size_t& dimensions) {
    std::unordered_map<std::string, int> documentFrequency;
    for (const Counts& doc : counts) {
        for (const auto& term : doc.terms) ++documentFrequency[term.first];
    }
    std::vector<std::pair<std::string, int>> vocabulary(documentFrequency.begin(),
                                                        documentFrequency.end());
    std::sort(vocabulary.begin(), vocabulary.end(), [](const auto& a, const auto& b) {
        return a.second != b.second ? a.second > b.second : a.first < b.first;
    });
    dimensions = std::min(requestedVocabulary, vocabulary.size());
    if (dimensions == 0) throw std::runtime_error("no terms remain after stop-word filtering");
    vocabulary.resize(dimensions);
    std::unordered_map<std::string, std::size_t> indices;
    for (std::size_t column = 0; column < dimensions; ++column) {
        indices[vocabulary[column].first] = column;
    }
    std::vector<double> vectors(counts.size() * dimensions, 0.0);
    for (std::size_t row = 0; row < counts.size(); ++row) {
        if (counts[row].total == 0) continue;
        double normSquared = 0.0;
        for (const auto& term : counts[row].terms) {
            const auto column = indices.find(term.first);
            if (column == indices.end()) continue;
            const double tf = static_cast<double>(term.second) / counts[row].total;
            const double idf = std::log((1.0 + counts.size()) /
                                         (1.0 + documentFrequency.at(term.first))) + 1.0;
            const double weight = tf * idf;
            vectors[row * dimensions + column->second] = weight;
            normSquared += weight * weight;
        }
        if (normSquared > 0.0) {
            const double norm = std::sqrt(normSquared);
            for (std::size_t column = 0; column < dimensions; ++column) {
                vectors[row * dimensions + column] /= norm;
            }
        }
    }
    return vectors;
}

double cpuDot(const std::vector<double>& vectors, std::size_t dimensions,
              std::size_t first, std::size_t second) {
    double dot = 0.0;
    for (std::size_t column = 0; column < dimensions; ++column) {
        dot += vectors[first * dimensions + column] * vectors[second * dimensions + column];
    }
    return dot;
}

std::vector<double> cpuCosine(const std::vector<double>& vectors,
                              std::size_t rows, std::size_t dimensions) {
    std::vector<double> similarity(rows * rows, 0.0);
#pragma omp parallel for schedule(dynamic)
    for (long long row = 0; row < static_cast<long long>(rows); ++row) {
        for (std::size_t column = 0; column < rows; ++column) {
            similarity[static_cast<std::size_t>(row) * rows + column] =
                cpuDot(vectors, dimensions, static_cast<std::size_t>(row), column);
        }
    }
    return similarity;
}

void verifyResults(const std::vector<double>& vectors, std::size_t rows,
                   std::size_t dimensions, const std::vector<double>& similarity) {
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < rows; ++column) {
            const double score = similarity[row * rows + column];
            if (!std::isfinite(score) || score < -1e-9 || score > 1.0 + 1e-9 ||
                std::abs(score - similarity[column * rows + row]) > 1e-9) {
                throw std::runtime_error("similarity is non-finite, out of range, or asymmetric");
            }
        }
    }
    const std::size_t sample = std::min<std::size_t>(rows, 32);
    for (std::size_t row = 0; row < sample; ++row) {
        for (std::size_t column = 0; column < sample; ++column) {
            const double expected = cpuDot(vectors, dimensions, row, column);
            if (!std::isfinite(similarity[row * rows + column]) ||
                std::abs(expected - similarity[row * rows + column]) > 1e-9) {
                throw std::runtime_error("similarity failed CPU reference check");
            }
        }
    }
}

std::string csvField(const std::string& value) {
    std::string result = "\"";
    for (char character : value) {
        if (character == '"') result += '"';
        result += character;
    }
    return result + '"';
}

void writeMatches(const Options& options, const std::vector<Movie>& movies,
                  const std::vector<double>& similarity) {
    std::vector<Match> matches;
    for (std::size_t row = 0; row < movies.size(); ++row) {
        for (std::size_t column = row + 1; column < movies.size(); ++column) {
            matches.push_back({row, column, similarity[row * movies.size() + column]});
        }
    }
    std::sort(matches.begin(), matches.end(), [](const Match& a, const Match& b) {
        if (a.score != b.score) return a.score > b.score;
        if (a.first != b.first) return a.first < b.first;
        return a.second < b.second;
    });
    const std::filesystem::path path(options.output);
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot write output: " + options.output);
    output << "movie_a,movie_b,cosine_similarity\n" << std::setprecision(12);
    for (std::size_t index = 0; index < std::min(options.top, matches.size()); ++index) {
        const Match& match = matches[index];
        output << csvField(movies[match.first].title) << ','
               << csvField(movies[match.second].title) << ',' << match.score << '\n';
    }
    if (!output) throw std::runtime_error("failed writing output: " + options.output);
}
} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int tasks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &tasks);
    try {
        const Options options = parseOptions(argc, argv);
        const double start = MPI_Wtime();
        const auto movies = readMovies(options);
        const auto stopwords = readStopwords(options.stopwords);
        const auto counts = gatherCounts(movies, stopwords, rank, tasks);
        if (rank == 0) {
            std::size_t dimensions = 0;
            const auto vectors = makeVectors(counts, options.vocabulary, dimensions);
            const double preprocessingSeconds = MPI_Wtime() - start;
            const double similarityStart = MPI_Wtime();
#ifdef USE_CUDA
            const auto similarity = cudaCosine(vectors, movies.size(), dimensions);
            const char* backend = "CUDA";
#else
            const auto similarity = cpuCosine(vectors, movies.size(), dimensions);
            const char* backend = "CPU";
#endif
            if (options.verify) verifyResults(vectors, movies.size(), dimensions, similarity);
            const double similaritySeconds = MPI_Wtime() - similarityStart;
            writeMatches(options, movies, similarity);
            std::cout << "Movies: " << movies.size() << ", MPI ranks: " << tasks
                      << ", OpenMP threads: " << omp_get_max_threads()
                      << ", vocabulary: " << dimensions << ", backend: " << backend << '\n'
                      << "Preprocessing: " << preprocessingSeconds << " s, similarity: "
                      << similaritySeconds << " s\n"
                      << "Wrote " << options.output << '\n';
            if (options.verify) std::cout << "CPU reference check passed\n";
        }
        MPI_Finalize();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Rank " << rank << ": " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
