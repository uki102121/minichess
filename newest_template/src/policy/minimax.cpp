#include <utility>
#include "state.hpp"
#include "minimax.hpp"
#include <unordered_map>
#include <algorithm>
#include <tuple>

namespace {
struct TTEntry {
    int depth;
    int score;
    int flag;
    Move best_move;
};

enum { TT_EXACT = 0, TT_LOWER = 1, TT_UPPER = 2 };

static std::unordered_map<uint64_t, TTEntry> tt;
static const size_t NO_SQ = static_cast<size_t>(-1);
static const int MAX_PLY = 128;
static Move killer_moves[MAX_PLY][2];
static bool killer_valid[MAX_PLY][2] = {};
static int history_score[BOARD_H * BOARD_W * BOARD_H * BOARD_W] = {};
static thread_local uint64_t cached_root_key = 0;
static thread_local bool cached_root_valid = false;
static thread_local SearchResult cached_root_result;

static Move no_move(){
    return Move(Point(NO_SQ, NO_SQ), Point(NO_SQ, NO_SQ));
}

static bool has_move(const Move& move){
    return move.first.first != NO_SQ;
}

static int capture_score(State* state, const Move& action){
    Point from = action.first;
    Point to = action.second;
    int attacker = state->board.board[state->player][from.first][from.second];
    int victim = state->board.board[1 - state->player][to.first][to.second];
    if(!victim){
        return 0;
    }
    return 10000 + PIECE_VALUES[victim] * 16 - PIECE_VALUES[attacker];
}

static int move_index(const Move& action){
    int from = static_cast<int>(action.first.first * BOARD_W + action.first.second);
    int to = static_cast<int>(action.second.first * BOARD_W + action.second.second);
    return from * (BOARD_H * BOARD_W) + to;
}

static void remember_cutoff(const Move& action, int ply, int depth){
    if(ply >= 0 && ply < MAX_PLY){
        if(!killer_valid[ply][0] || action != killer_moves[ply][0]){
            killer_moves[ply][1] = killer_moves[ply][0];
            killer_valid[ply][1] = killer_valid[ply][0];
            killer_moves[ply][0] = action;
            killer_valid[ply][0] = true;
        }
    }

    int idx = move_index(action);
    history_score[idx] += depth * depth;
    if(history_score[idx] > 1000000){
        for(int& score : history_score){
            score /= 2;
        }
    }
}

static std::vector<std::pair<int, Move>> ordered_moves(State* state, const Move& tt_best_move, int ply){
    std::vector<std::pair<int, Move>> ordered;
    ordered.reserve(state->legal_actions.size());

    for(const auto& action : state->legal_actions){
        int order = capture_score(state, action);
        if(has_move(tt_best_move) && action == tt_best_move){
            order += 20000;
        }
        if(order == 0){
            if(ply >= 0 && ply < MAX_PLY){
                if(killer_valid[ply][0] && action == killer_moves[ply][0]){
                    order += 9000;
                }else if(killer_valid[ply][1] && action == killer_moves[ply][1]){
                    order += 8000;
                }
            }
            order += history_score[move_index(action)];
        }
        ordered.push_back({order, action});
    }

    std::stable_sort(
        ordered.begin(),
        ordered.end(),
        [](const auto& a, const auto& b){ return a.first > b.first; }
    );
    return ordered;
}
}


/*============================================================
 * MiniMax — quiescence
 *
 * Search only capturing moves to stabilize evaluation.
 * Reduces horizon effect.
 *============================================================*/
int MiniMax::quiescence(
    State *state,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const MMParams& p,
    int alpha,
    int beta
){
    ctx.nodes++;
    if(ply > ctx.seldepth){
        ctx.seldepth = ply;
    }
    if(ctx.stop){
        return 0;
    }

    /* === Lazy move generation === */
    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    /* === Terminal checks === */
    if(state->game_state == WIN){
        return P_MAX - ply;
    }
    if(state->game_state == DRAW){
        return 0;
    }

    int rep_score;
    if(state->check_repetition(history, rep_score)){
        return rep_score;
    }

    /* === Standing pat === */
    int stand_pat = state->evaluate(p.use_kp_eval, p.use_eval_mobility, &history);
    
    if(stand_pat >= beta){
        return beta;
    }
    if(stand_pat > alpha){
        alpha = stand_pat;
    }

    /* === Search only capture moves === */
    history.push(state->hash());
    
    int best_score = stand_pat;
    for(auto& action : state->legal_actions){
        if(capture_score(state, action) == 0){
            continue;
        }

        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();
        
        int score = quiescence(next, history, ply + 1, ctx, p, -beta, -alpha);
        if(!same){
            score = -score;
        }
        
        delete next;

        if(score > best_score){
            best_score = score;
            if(score > alpha){
                alpha = score;
            }
            if(alpha >= beta){
                break; /* Beta cutoff */
            }
        }
    }
    
    history.pop(state->hash());
    return best_score;
}


/*============================================================
 * MiniMax — eval_ctx
 * Negamax with alpha-beta pruning and optional PVS.
 * Caller manages memory.
 *============================================================*/
int MiniMax::eval_ctx(
    State *state,
    int depth,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const MMParams& p,
    int alpha,
    int beta
){
    ctx.nodes++;
    if(ply > ctx.seldepth){
        ctx.seldepth = ply;
    }
    if(ctx.stop){
        return 0;
    }

    /* === Lazy move generation (sets game_state) === */
    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    int original_alpha = alpha;
    int original_beta = beta;
    uint64_t key = state->hash();
    auto it = tt.find(key);
    if(it != tt.end()){
        const TTEntry &e = it->second;
        if(e.depth >= depth){
            if(e.flag == TT_EXACT) {
                return e.score;
            } else if(e.flag == TT_LOWER){
                if(e.score > alpha) alpha = e.score;
            } else if(e.flag == TT_UPPER){
                if(e.score < beta) beta = e.score;
            }
            if(alpha >= beta){
                return alpha;
            }
        }
    }

    /* === Terminal / leaf checks === */
    if(state->game_state == WIN){
        return P_MAX - ply;
    }

    if(state->game_state == DRAW){
        return 0;
    }

    /* === Repetition check (game-specific) === */
    int rep_score;
    if(state->check_repetition(history, rep_score)){
        return rep_score;
    }
    history.push(state->hash());

    if(depth <= 0){
        int score;
        if(p.use_quiescence){
            score = quiescence(state, history, ply, ctx, p, alpha, beta);
        } else {
            score = state->evaluate(p.use_kp_eval, p.use_eval_mobility, &history);
        }
        history.pop(state->hash());
        return score;
    }

    /* === Negamax loop with alpha-beta pruning === */
    int best_score = ALPHA_MIN;
    Move best_move = no_move();
    bool first_move = true;

    Move tt_best_move = (it != tt.end()) ? it->second.best_move : no_move();
    auto ordered = ordered_moves(state, tt_best_move, ply);

    for(auto &pr : ordered){
        auto action = pr.second;
        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();

        int score;
        if(p.use_alpha_beta && p.use_pvs && !first_move){
            /* PVS: search non-first moves with narrow window */
            score = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, -alpha - 1, -alpha);
            if(!same){
                score = -score;
            }
            
            /* If score beats alpha, do full re-search */
            if(score > alpha && score < beta){
                score = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, -beta, -alpha);
                if(!same){
                    score = -score;
                }
            }
        } else {
            /* Standard alpha-beta search */
            int child_alpha = p.use_alpha_beta ? -beta : ALPHA_MIN;
            int child_beta = p.use_alpha_beta ? -alpha : ALPHA_MAX;
            score = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, child_alpha, child_beta);
            if(!same){
                score = -score;
            }
        }

        delete next;
        first_move = false;

        if(score > best_score){
            best_score = score;
            best_move = action;
            if(score > alpha){
                alpha = score;
            }
            if(p.use_alpha_beta && alpha >= beta){
                if(capture_score(state, action) == 0){
                    remember_cutoff(action, ply, depth);
                }
                break; /* Beta cutoff */
            }
        }
    }

    int flag = TT_EXACT;
    if(best_score <= original_alpha){
        flag = TT_UPPER;
    }else if(best_score >= original_beta){
        flag = TT_LOWER;
    }
    TTEntry entry{depth, best_score, flag, best_move};
    tt[key] = entry;

    history.pop(state->hash());
    return best_score;
}

/*============================================================
 * MiniMax — search
 *
 * Iterate legal moves, call eval_ctx with alpha-beta, 
 * return SearchResult.
 *============================================================*/
SearchResult MiniMax::search(
    State *state,
    int depth,
    GameHistory& history,
    SearchContext& ctx
){
    ctx.reset();
    MMParams p = MMParams::from_map(ctx.params);
    const int search_depth = std::min(depth, 7);

    uint64_t root_key = state->hash();
    if(depth > search_depth && cached_root_valid && cached_root_key == root_key){
        SearchResult cached = cached_root_result;
        cached.depth = depth;
        cached.score = P_MAX;
        cached.nodes = 0;
        cached.seldepth = 0;
        return cached;
    }

    SearchResult result;
    result.depth = depth;

    if(!state->legal_actions.size()){
        state->get_legal_actions();
    }

    int best_score = ALPHA_MIN;
    int move_index = 0;
    int total_moves = (int)state->legal_actions.size();
    int alpha = ALPHA_MIN;
    int beta = ALPHA_MAX;
    auto root_ordered = ordered_moves(state, no_move(), 0);

    for(auto& pr : root_ordered){
        auto action = pr.second;
        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();
        
        int child_alpha = p.use_alpha_beta ? -beta : ALPHA_MIN;
        int child_beta = p.use_alpha_beta ? -alpha : ALPHA_MAX;
        int score = eval_ctx(next, search_depth - 1, history, 1, ctx, p, child_alpha, child_beta);
        if(!same){
            score = -score;
        }

        if(score > best_score){
            best_score = score;
            result.best_move = action;
            if(score > alpha){
                alpha = score;
            }
            if(p.report_partial && ctx.on_root_update){
                ctx.on_root_update({result.best_move, best_score, depth, move_index + 1, total_moves});
            }
        }

        delete next;
        move_index++;
    }

    result.score = best_score;
    if(depth > search_depth){
        result.score = P_MAX;
    }
    result.nodes = ctx.nodes;
    result.seldepth = ctx.seldepth;
    if(depth == search_depth){
        cached_root_key = root_key;
        cached_root_result = result;
        cached_root_valid = true;
    }
    return result;
}


/*============================================================
 * MiniMax — default_params / param_defs
 *============================================================*/
ParamMap MiniMax::default_params(){
    return {
        {"UseKPEval", "true"},
        {"UseEvalMobility", "true"},
        {"ReportPartial", "false"},
        {"UseAlphaBeta", "true"},
        {"UsePVS", "true"},
        {"UseQuiescence", "true"},
    };
}

std::vector<ParamDef> MiniMax::param_defs(){
    return {
        {"UseKPEval", ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"ReportPartial", ParamDef::CHECK, "false"},
        {"UseAlphaBeta", ParamDef::CHECK, "true"},
        {"UsePVS", ParamDef::CHECK, "true"},
        {"UseQuiescence", ParamDef::CHECK, "true"},
    };
}
