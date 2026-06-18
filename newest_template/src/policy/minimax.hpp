#pragma once
#include "search_types.hpp"
#include "game_history.hpp"

/* alpha-beta bounds local to policy layer (do not modify base_state.hpp) */
constexpr int ALPHA_MIN = -100001;
constexpr int ALPHA_MAX = 100001;

struct MMParams {
    bool use_kp_eval = true;
    bool use_eval_mobility = true;
    bool report_partial = true;
    bool use_alpha_beta = true;
    bool use_pvs = false;
    bool use_quiescence = true;

    static MMParams from_map(const ParamMap& m){
        MMParams p;
        p.use_kp_eval       = param_bool(m, "UseKPEval", true);
        p.use_eval_mobility = param_bool(m, "UseEvalMobility", true);
        p.report_partial    = param_bool(m, "ReportPartial", true);
        p.use_alpha_beta    = param_bool(m, "UseAlphaBeta", true);
        p.use_pvs           = param_bool(m, "UsePVS", false);
        p.use_quiescence    = param_bool(m, "UseQuiescence", true);
        return p;
    }
};

class MiniMax{
public:
    static int eval_ctx(
        State *state,
        int depth,
        GameHistory& history,
        int ply,
        SearchContext& ctx,
        const MMParams& p,
        int alpha = ALPHA_MIN,
        int beta = ALPHA_MAX
    );
    
    static int quiescence(
        State *state,
        GameHistory& history,
        int ply,
        SearchContext& ctx,
        const MMParams& p,
        int alpha = ALPHA_MIN,
        int beta = ALPHA_MAX
    );
    
    static SearchResult search(
        State *state,
        int depth,
        GameHistory& history,
        SearchContext& ctx
    );

    static ParamMap default_params();
    static std::vector<ParamDef> param_defs();
};
